"""Feature engineering for daily demand forecasting.  ONE function, used by EVERYTHING:
training, evaluation, the weekly batch forecaster, drift checks and the real-time API.
(One implementation = no training-serving skew.  tests/test_history_store.py proves the parity.)

THE GOLDEN RULE
---------------
We stand at the END of day D and forecast day D+1.  A feature for the row "day t" may therefore use
sales of days t-1, t-2, ... but NEVER sales of day t or later.

How lags and rolling means are built (per series, on a COMPLETE daily calendar)
-------------------------------------------------------------------------------
    lag_k            = sales.shift(k)                       # the value k days earlier
    rolling_mean_w   = sales.shift(1).rolling(w).mean()     # shift(1) FIRST, then roll

Why shift(1) BEFORE rolling?  rolling(7) on the raw column includes today.  Shifting first slides the
window so it covers days t-7 ... t-1.

Worked example (one series, sales for days 1..8):  [10, 12, 11, 13, 20, 22, 21, 14]
    day 8, rolling_mean_7 WITHOUT shift  = mean(12,11,13,20,22,21,14) = 16.14  <- includes day 8 itself (leak!)
    day 8, rolling_mean_7 WITH shift(1)  = mean(10,12,11,13,20,22,21) = 15.57  <- days 1..7 only (correct)
    day 8, lag_1 = 21, lag_7 = 10 (day 1)

Other features
--------------
  price_change   = price / price_yesterday - 1            e.g. 38 / 40 - 1 = -0.05 (a 5% price cut starts today)
  discount_pct   = discount / (price + discount)          e.g. 2 / (38 + 2) = 0.05  (share of the list price given away)
  dow_sin/cos    = sin/cos(2*pi*day_of_week/7)            cyclical encoding: Sunday (6) is next to Monday (0)
                   e.g. Monday  -> (0.000, 1.000);  Thursday (3) -> (0.434, -0.901);  Sunday (6) -> (-0.782, 0.623)
  trend_index    = days since 2023-01-01                   lets a linear model learn a slope

Missing values
--------------
  * a missing calendar day (POS outage) becomes a NaN sales row (we re-index to a complete calendar), so
    shift(1) always means "the previous CALENDAR day", not "the previous row".
  * if lag_k is NaN we fill it with that day's rolling_mean_7 (which is known at prediction time).
  * rolling means need only a few valid days (min_periods) so a single outage does not kill the feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import DATA_RAW, FEATURES_PATH
from src.features.definitions import KEYS, TREND_ORIGIN, WARMUP_DAYS

REQUIRED = ["date", "product_id", "store_id", "sales", "price", "discount", "promotion", "holiday", "forecast_temp"]


def check_history(df: pd.DataFrame) -> None:
    """Fail loudly on broken input.  Duplicate (date, product, store) rows - e.g. after a pipeline RETRY -
    would silently shift every rolling window, so we refuse to continue."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    dup = int(df.duplicated(["date", *KEYS]).sum())
    if dup:
        raise ValueError(f"{dup} duplicated (date, product_id, store_id) rows - did a pipeline step run twice?")


def complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a NaN row for every missing day inside each series, so that shift(1) = 'yesterday'."""
    parts = []
    for (p, s), g in df.groupby(KEYS, sort=False):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").drop(columns=KEYS).reindex(full)
        g.index.name = "date"
        g["product_id"], g["store_id"] = p, s
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def _rolling_mean(shifted: pd.Series, d: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """Rolling mean of an ALREADY SHIFTED series, computed separately inside each series."""
    out = shifted.groupby([d["product_id"], d["store_id"]], sort=False).rolling(window, min_periods=min_periods).mean()
    return out.reset_index(level=[0, 1], drop=True).reindex(shifted.index)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw daily rows (+ optionally future rows with sales = NaN) -> the same rows with all features.

    Input columns : date, product_id, store_id, sales, price, discount, promotion, holiday, forecast_temp
    The output is sorted by (product_id, store_id, date) and has the extra columns
    history_days (days since the series started in this frame) and every feature in
    definitions.ALL_CANDIDATE_FEATURES.
    """
    check_history(df)
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values([*KEYS, "date"]).reset_index(drop=True)
    d = complete_calendar(d)
    d = d.sort_values([*KEYS, "date"]).reset_index(drop=True)
    g = d.groupby(KEYS, sort=False)

    # ---- sales history: every feature below only looks at days BEFORE the row's date ----------
    sales_yesterday = g["sales"].shift(1)
    d["lag_1"] = sales_yesterday
    d["lag_7"] = g["sales"].shift(7)
    d["lag_14"] = g["sales"].shift(14)
    d["rolling_mean_7"] = _rolling_mean(sales_yesterday, d, 7, 4)
    d["rolling_mean_30"] = _rolling_mean(sales_yesterday, d, 30, 15)
    for c in ("lag_1", "lag_7", "lag_14"):                       # a missing day: fall back to the recent average
        d[c] = d[c].fillna(d["rolling_mean_7"])
    d["rolling_mean_30"] = d["rolling_mean_30"].fillna(d["rolling_mean_7"])
    d["history_days"] = (d["date"] - g["date"].transform("min")).dt.days

    # ---- calendar (known in advance) ---------------------------------------------------------
    dt = d["date"].dt
    d["day_of_week"] = dt.dayofweek
    d["week_of_year"] = dt.isocalendar().week.astype(int).values
    d["month"] = dt.month
    d["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    d["is_holiday"] = d["holiday"].fillna(0).astype(int)
    d["trend_index"] = (d["date"] - TREND_ORIGIN).dt.days
    d["dow_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    d["month_sin"] = np.sin(2 * np.pi * (d["month"] - 1) / 12)
    d["month_cos"] = np.cos(2 * np.pi * (d["month"] - 1) / 12)

    # ---- price & promotion (planned, so known in advance) -----------------------------------------
    price_yesterday = g["price"].shift(1)
    d["price_change"] = (d["price"] / price_yesterday - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["discount_pct"] = (d["discount"] / (d["price"] + d["discount"])).fillna(0.0)
    d["promotion_flag"] = d["promotion"].fillna(0).astype(int)
    return d


def features_for_batch(batch: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Offline, point-in-time-correct features for a labelled production batch.

    batch   : production batch (BATCH_COLUMNS or raw layout, with the TRUE sales)
    history : earlier raw rows (training data).  Only rows BEFORE the first batch date are used.
    Row t of the batch gets lags from days < t (batch actuals included) - exactly what the nightly
    feature refresh would have stored.
    """
    b = batch.rename(columns={"forecast_date": "date"}).copy()
    b["date"] = pd.to_datetime(b["date"])
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h[h["date"] < b["date"].min()]
    both = pd.concat([h[REQUIRED], b[REQUIRED]], ignore_index=True)
    f = compute_features(both)
    return f[f["date"] >= b["date"].min()].reset_index(drop=True)


def training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineered frame ready for modelling: warm-up rows (no full 30-day history) and rows with an
    unknown target (POS outage) are dropped."""
    f = compute_features(raw)
    return f[(f["history_days"] >= WARMUP_DAYS) & f["sales"].notna()].reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    feats = training_frame(raw)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(FEATURES_PATH, index=False)
    print(f"engineered frame: {feats.shape[0]:,} rows x {feats.shape[1]} columns -> {FEATURES_PATH.name}")
