"""THE WEEKLY JOB (every Monday):  1) refresh the history store with last week's actual sales,  2) forecast the next 7 days.

    python -m src.models.weekly_job --new-actuals data/production/batch_drifted.csv --through 2025-01-07
        [--history data/raw/sales_daily.csv] [--out reports/history_refreshed.csv]

NO new model is trained here - the model only changes in the monthly / triggered retrain (src/models/retrain.py).
But the INPUTS change every week: the lags and rolling means must see last week's sales, otherwise forecasts go stale
(failure scenario (a), see src/monitoring/failure_scenarios.py).

IDEMPOTENT refresh: a pipeline that is retried must not insert last week twice (duplicated dates inflate rolling means).
`upsert_history` therefore replaces rows with the same (date, product_id, store_id) instead of appending blindly.
Worked example: history has 100 rows, new batch has 7 rows of which 3 already exist -> 4 new + 3 replaced -> 104 rows.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, DATA_RAW, REPORTS
from src.data.make_dataset import from_batch
from src.features.build_features import check_history
from src.features.definitions import HISTORY_COLUMNS, KEYS
from src.models.batch_forecast import recursive_forecast


def upsert_history(history: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add new actuals to the history.  Same (date, product, store) -> the NEW row wins.  Result has no duplicates."""
    h = history[HISTORY_COLUMNS].copy()
    n = new_rows[HISTORY_COLUMNS].copy()
    h["date"], n["date"] = pd.to_datetime(h["date"]), pd.to_datetime(n["date"])
    key = ["date", *KEYS]
    existing = h.set_index(key).index
    replaced = int(n.set_index(key).index.isin(existing).sum())
    out = pd.concat([h, n]).drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    check_history(out)
    return out, dict(rows_before=len(h), new=len(n) - replaced, replaced=replaced, rows_after=len(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(DATA_RAW / "sales_daily.csv"))
    ap.add_argument("--new-actuals", required=True, help="production batch with the TRUE sales of the finished days")
    ap.add_argument("--through", required=True, help="last finished day (YYYY-MM-DD); later rows are ignored")
    ap.add_argument("--out", default=str(REPORTS / "history_refreshed.csv"))
    a = ap.parse_args()
    hist = pd.read_csv(a.history, parse_dates=["date"])
    new = from_batch(pd.read_csv(a.new_actuals))
    new = new[(new["date"] <= pd.Timestamp(a.through)) & new["sales"].notna()]
    print(f"WEEKLY JOB as of {a.through}")
    print(f"1) FEATURE REFRESH: history ends {hist['date'].max().date()}, new actuals {new['date'].min().date()}..{new['date'].max().date()} ({len(new)} rows)")
    refreshed, info = upsert_history(hist, new)
    print(f"   first run : {info}")
    again, info2 = upsert_history(refreshed, new)
    print(f"   RE-RUN (simulated pipeline retry): {info2}  -> still {len(again)} rows, no duplicates (idempotent)")
    refreshed.to_csv(a.out, index=False)
    art = load_artifact(CHAMPION_PATH)
    f = recursive_forecast(refreshed, art, pd.Timestamp(a.through), 7)
    REPORTS.mkdir(exist_ok=True)
    f.to_csv(REPORTS / "forecast_next_7d.csv", index=False)
    print(f"2) BATCH FORECAST with champion {art['version']}: {len(f)} rows (24 series x 7 days) -> reports/forecast_next_7d.csv")
    print(f[f.product_id.eq("P04") & f.store_id.eq("S2")][["forecast_date", "horizon", "predicted_units", "lower", "upper"]].to_string(index=False))


if __name__ == "__main__":
    main()
