"""Data profile: look at the data BEFORE modelling.  (Prints what docs/DATASET.md describes.)

Run:  python -m src.data.profile_data

What to look for (the questions a data scientist asks first)
  1. How big is it, which dates, how many series?
  2. Is the target sane?  zeros (stock-outs?), huge values (outliers?), missing days?
  3. Which columns have missing values and WHY (a missing weather forecast is not the same as a missing sale)?
  4. Is there TREND?  The weekly-average table answers that: level 100 -> 130 in training, 150 -> 210 in production.
  5. Which features are strongly correlated with the target - and with each other (redundancy)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.facts import save_fact
from src.common.paths import DATA_PRODUCTION, DATA_RAW
from src.data.make_dataset import DRIFT_STEPS, NORMAL_KNOTS, TRAIN_KNOTS, level_path
from src.features.build_features import training_frame
from src.features.definitions import ALL_CANDIDATE_FEATURES


def weekly_average_table(train: pd.DataFrame, normal: pd.DataFrame, drifted: pd.DataFrame) -> pd.DataFrame:
    """Average units per series-day in selected 7-day weeks + the design 'level' used by the generator."""
    rows = []

    def add(label, df, start, knots, steps=False):
        w = df[(df["date"] >= start) & (df["date"] < start + pd.Timedelta(days=7))]
        lvl = float(level_path(pd.DatetimeIndex([start + pd.Timedelta(days=3)]), knots, steps)[0])
        rows.append(dict(period=label, week_start=start.date(), avg_units_per_series_day=round(float(w["sales"].mean()), 1),
                         design_level=round(lvl, 0)))

    t0 = train["date"].min()
    for k in (0, 26, 52, 78, 104):
        add(f"training week {k + 1}", train, t0 + pd.Timedelta(weeks=k), TRAIN_KNOTS)
    p0 = normal["date"].min()
    for k in range(8):
        add(f"prod NORMAL week {k + 1}", normal, p0 + pd.Timedelta(weeks=k), NORMAL_KNOTS)
    for k in range(8):
        add(f"prod DRIFTED week {k + 1}", drifted, p0 + pd.Timedelta(weeks=k), DRIFT_STEPS, True)
    return pd.DataFrame(rows)


def main() -> None:
    train = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    normal = pd.read_csv(DATA_PRODUCTION / "batch_normal.csv", parse_dates=["forecast_date"]).rename(columns={"forecast_date": "date"})
    drifted = pd.read_csv(DATA_PRODUCTION / "batch_drifted.csv", parse_dates=["forecast_date"]).rename(columns={"forecast_date": "date"})
    pd.set_option("display.width", 160)

    print("=== 1. SHAPE ===")
    print(f"training rows      : {len(train):,}  ({train.date.min().date()} .. {train.date.max().date()}, {train.date.nunique()} days)")
    print(f"series             : {train.groupby(['product_id', 'store_id']).ngroups}  "
          f"({train.product_id.nunique()} products x {train.store_id.nunique()} stores)")
    print(f"production rows    : normal={len(normal):,}  drifted={len(drifted):,}  ({normal.date.min().date()} .. {normal.date.max().date()})")
    print("\nfirst 4 rows:\n" + train.head(4).to_string(index=False))

    print("\n=== 2. TARGET: units sold per series-day ===")
    s = train["sales"]
    print(s.describe().round(1).to_string())
    zeros = int((s == 0).sum())
    print(f"\nzero-sales rows (stock-outs)         : {zeros} ({zeros / len(train):.2%})  <- MAPE would divide by zero here")
    med14 = train.groupby(["product_id", "store_id"])["sales"].transform(lambda x: x.rolling(15, center=True, min_periods=5).median())
    out = int((s > 2.2 * med14).sum())
    print(f"candidate outliers (> 2.2 x local median): {out} ({out / len(train):.2%})  <- bulk orders / data-entry spikes")

    print("\n=== 3. MISSING VALUES (injected, documented in docs/DATASET.md) ===")
    miss = train.isna().mean()
    print(miss[miss > 0].map(lambda v: f"{v:.2%}").to_string())
    print("  sales missing        = POS outage of a whole store-day        -> MCAR (independent of the sales value)")
    print("  forecast_temp missing = weather-API outage of a whole store-day -> MCAR; handled by median + missing indicator")
    print("  temperature (ACTUAL)  = never missing but NOT available at prediction time -> rejected (leakage)")

    print("\n=== 4. TREND: weekly average sales (units per series-day) vs the design level ===")
    wt = weekly_average_table(train, normal, drifted)
    print(wt.to_string(index=False))
    print("Averages sit a little below the design level in Jan-Feb (winter dip, fewer cold-drink sales); the SHIFT is clear:")
    print("  training level 100 -> 130 over two years;  DRIFTED production 150 -> 170 -> 190 -> 210 in four steps.")

    print("\n=== 5. WEEKLY SEASONALITY (average units by weekday, training) ===")
    dow = train.groupby("day_of_week")["sales"].mean()
    print((dow / dow.mean()).round(2).rename(index=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))).to_string())

    print("\n=== 6. CORRELATIONS (engineered training frame) ===")
    f = training_frame(train)
    num = f[[c for c in ALL_CANDIDATE_FEATURES if c not in ("product_id", "store_id")] + ["sales"]]
    c_target = num.corr()["sales"].drop("sales").sort_values(key=abs, ascending=False)
    print("top |correlation| with sales:\n" + c_target.head(8).round(3).to_string())
    print("NOTE: price correlates strongly NEGATIVELY with sales only because expensive products (rice, shampoo) simply sell\n"
          "      fewer units than cheap ones (a BETWEEN-product effect).  It is NOT 'raising the price cuts sales' -\n"
          "      correlation is not causation.  That is one reason absolute price is not used as a model input.")
    cm = num.drop(columns="sales").corr().abs()
    pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(cm.columns) for b in cm.columns[i + 1:]]
    pairs = sorted(pairs, key=lambda x: -x[2])[:6]
    print("\nmost correlated FEATURE pairs (redundancy):")
    for a, b, v in pairs:
        print(f"  {a:<16} {b:<16} |r|={v:.3f}")

    save_fact("data_profile", dict(
        train_shape=list(train.shape), production_rows=dict(normal=len(normal), drifted=len(drifted)), series=24,
        date_range=[str(train.date.min().date()), str(train.date.max().date())],
        target_rate=dict(mean_units=round(float(s.mean()), 2), zero_rate=round(zeros / len(train), 4)),
        missing_rates={k: round(float(v), 4) for k, v in miss[miss > 0].items()},
        top_correlations_with_target=c_target.head(8).round(3).to_dict(),
        top_feature_pairs=[dict(a=a, b=b, abs_r=round(float(v), 3)) for a, b, v in pairs],
        weekly_average_table=wt.astype(str).to_dict("records"), outlier_candidates=out, zero_rows=zeros))


if __name__ == "__main__":
    main()
