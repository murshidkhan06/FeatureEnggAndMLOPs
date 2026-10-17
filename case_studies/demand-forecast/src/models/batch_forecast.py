"""BATCH serving (primary): every Monday forecast the next 7 days for all 24 series -> reports/forecast_next_7d.csv

    python -m src.models.batch_forecast                      # as of the last day in data/processed/sales_history.csv
    python -m src.models.batch_forecast --backtest           # how good are 1..7 day-ahead forecasts? (weekly origins, test period)

BATCH vs REAL-TIME
  batch      : one job, all series at once, results written to a file/table.  Used for WEEKLY PLANNING (orders, staffing).
               Cheap, easy to monitor, can be re-run.
  real-time  : POST /predict for one product-store-date on demand (replenishment screen, "what if I run a promotion?").
               Needs a history store and low latency.
  Same model, same compute_features() - only the plumbing differs.

RECURSIVE (iterative) MULTI-STEP FORECAST - explained simply
  Our model forecasts ONE day ahead because its features are yesterday's numbers.  To forecast 7 days we walk forward:
     day D+1 : lags are real sales            -> forecast F1
     day D+2 : lag_1 = F1  (our own forecast!) -> forecast F2
     day D+3 : lag_1 = F2, lag_2 ...          -> forecast F3  ... and so on up to D+7.
  The leakage rule still holds (only data up to day D is real); but errors ACCUMULATE: the further ahead, the more of
  the lag features are our own forecasts, so WAPE grows with the horizon (see --backtest).

ASSUMPTIONS for the future days (no promotion plan is available in this demo)
  price = last list price, discount = 0, promotion = 0, holiday = calendar, forecast_temp = last-7-day average of the store.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.artifact import load_artifact
from src.common.facts import save_fact
from src.common.paths import CHAMPION_PATH, DATA_RAW, HISTORY_PATH, REPORTS
from src.features.build_features import REQUIRED, compute_features
from src.features.calendar import is_holiday
from src.features.definitions import KEYS
from src.models import metrics as M
from src.models.train import score


def recursive_forecast(history: pd.DataFrame, art: dict, as_of: pd.Timestamp, horizon: int = 7, tail_days: int = 60) -> pd.DataFrame:
    """history: rows with date <= as_of (extra rows are ignored).  Returns one row per series x horizon day."""
    h = history[pd.to_datetime(history["date"]) <= as_of][REQUIRED].copy()
    h["date"] = pd.to_datetime(h["date"])
    recent = h[h["date"] > as_of - pd.Timedelta(days=tail_days)]
    last = recent.sort_values("date").groupby(KEYS).tail(1).set_index(KEYS).sort_index()
    temp = recent[recent["date"] > as_of - pd.Timedelta(days=7)].groupby("store_id")["forecast_temp"].mean()
    works = recent.copy()
    out = []
    for step in range(1, horizon + 1):
        day = as_of + pd.Timedelta(days=step)
        new = last.reset_index().assign(date=day, sales=np.nan, discount=0.0, promotion=0, holiday=is_holiday(day))
        new["price"] = last["price"].values + last["discount"].values            # back to the LIST price
        new["forecast_temp"] = new["store_id"].map(temp).values
        feats = compute_features(pd.concat([works, new[REQUIRED]], ignore_index=True))
        rows = feats[feats["date"] == day].reset_index(drop=True)
        pred = score(art["pipeline"], rows, art["input_columns"])
        q = art["extra"]["interval"]
        width = np.sqrt(step)                                                    # rule of thumb: uncertainty grows with sqrt(horizon)
        out.append(pd.DataFrame(dict(product_id=rows.product_id, store_id=rows.store_id, forecast_date=day.date(), horizon=step,
                                     predicted_units=pred.round(1), lower=np.clip(pred * (1 + q["low"] * width), 0, None).round(1),
                                     upper=(pred * (1 + q["high"] * width)).round(1))))
        done = rows[KEYS].assign(date=day, sales=pred).merge(new[REQUIRED].drop(columns="sales"), on=["date", *KEYS])   # keyed merge: no order mix-ups
        works = pd.concat([works, done[REQUIRED]], ignore_index=True)   # our forecast becomes tomorrow's lag_1
    return pd.concat(out, ignore_index=True)


def backtest(raw: pd.DataFrame, art: dict, start: str, end: str, horizon: int = 7) -> pd.DataFrame:
    """Forecast `horizon` days from every Sunday origin in [start, end]; compare with what really sold."""
    raw = raw.assign(date=pd.to_datetime(raw["date"]))
    origins = [d for d in pd.date_range(start, end, freq="W-SUN") if d + pd.Timedelta(days=horizon) <= raw["date"].max()]
    rows = []
    for o in origins:
        f = recursive_forecast(raw, art, o, horizon)
        f["forecast_date"] = pd.to_datetime(f["forecast_date"])
        act = raw[["date", "product_id", "store_id", "sales"]].rename(columns={"date": "forecast_date"})
        m = f.merge(act, on=["forecast_date", "product_id", "store_id"]).dropna(subset=["sales"])
        # seasonal naive for the same days: sales exactly 7 (or 14, 21 ...) days earlier but not after the origin
        hist = raw.set_index(["date", "product_id", "store_id"])["sales"]
        lagdays = ((m["forecast_date"] - o).dt.days - 1) // 7 * 7 + 7
        m["seasonal_naive"] = [hist.get((d - pd.Timedelta(days=int(k)), p, s), np.nan) for d, p, s, k in zip(m.forecast_date, m.product_id, m.store_id, lagdays)]
        m["origin"] = o
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(HISTORY_PATH))
    ap.add_argument("--as-of", default=None, help="last day of real data (default: last day in the history file)")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    art = load_artifact(CHAMPION_PATH)
    pd.set_option("display.width", 160)

    if a.backtest:
        raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
        bt = backtest(raw, art, "2024-10-06", "2024-12-22", a.horizon)
        g = bt.groupby("horizon")
        tab = pd.DataFrame(dict(rows=g.size(), wape=g.apply(lambda x: M.wape(x.sales, x.predicted_units), include_groups=False),
                                bias_pct=g.apply(lambda x: M.bias_pct(x.sales, x.predicted_units), include_groups=False),
                                seasonal_naive_wape=g.apply(lambda x: M.wape(x.sales[x.seasonal_naive.notna()], x.seasonal_naive.dropna()), include_groups=False),
                                interval_coverage=g.apply(lambda x: float(((x.sales >= x.lower) & (x.sales <= x.upper)).mean()), include_groups=False)))
        print(f"BACKTEST: {bt.origin.nunique()} weekly origins ({bt.origin.min().date()} .. {bt.origin.max().date()}), "
              f"champion {art['version']} ({art['extra']['model_name']}), 24 series, recursive forecast")
        print(tab.round(3).to_string())
        ok7 = bool((tab.wape < tab.seasonal_naive_wape).all())
        print("\nReading: WAPE tends to rise with the horizon because lag features become our own forecasts (errors accumulate).")
        print(f"Horizon 1 is WORSE than the test-set WAPE of the champion because this batch job does not know the future: it assumes list price, no promotion")
        print("and an average temperature for the coming days (see the ASSUMPTIONS above).  The real-time API gets the real plan in the request.")
        print(f"Does the champion beat the seasonal-naive baseline at every horizon? {'YES' if ok7 else 'NO'} - if not, a plain 'same weekday last week' rule would be enough.")
        print("Interval coverage: the band is calibrated for horizon 1 on validation data and widened by sqrt(horizon); coverage below 80% at horizon 1 means it is a bit narrow.")
        REPORTS.mkdir(exist_ok=True)
        tab.to_csv(REPORTS / "horizon_wape.csv")
        save_fact("batch_backtest", tab.round(4).reset_index().to_dict("records"))
        return

    hist = pd.read_csv(a.history, parse_dates=["date"])
    as_of = pd.Timestamp(a.as_of) if a.as_of else hist["date"].max()
    f = recursive_forecast(hist, art, as_of, a.horizon)
    REPORTS.mkdir(exist_ok=True)
    f.to_csv(REPORTS / "forecast_next_7d.csv", index=False)
    print(f"Forecast made at the END of {as_of.date()} (all history <= that day) with champion {art['version']} ({art['extra']['model_name']})")
    print(f"{len(f)} rows = 24 series x {a.horizon} days -> reports/forecast_next_7d.csv\n")
    last7 = hist[(hist.date > as_of - pd.Timedelta(days=7)) & (hist.date <= as_of)].groupby("product_id")["sales"].sum()
    nxt = f.groupby("product_id")[["predicted_units", "lower", "upper"]].sum().round(0)
    nxt.insert(0, "last_7_days_actual", last7.round(0))
    print("NEXT 7 DAYS, total units per product (all 3 stores):")
    print(nxt.to_string())
    print("\nDaily forecast for one series (P04 milk, S2):")
    print(f[(f.product_id == "P04") & (f.store_id == "S2")][["forecast_date", "horizon", "predicted_units", "lower", "upper"]].to_string(index=False))
    save_fact("batch_forecast", dict(as_of=str(as_of.date()), rows=len(f), per_product=nxt.reset_index().to_dict("records")))


if __name__ == "__main__":
    main()
