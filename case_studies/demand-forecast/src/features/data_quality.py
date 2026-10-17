"""Data-quality checks - step 1 of the retraining decision ladder.

WHY first?  If the incoming data is broken (duplicates after a pipeline retry, prices suddenly in paise,
a store that stopped reporting) then ANY drift or accuracy number computed from it is meaningless.
The right reaction is "fix the pipeline", NOT "retrain the model" - retraining on broken data bakes the
damage into the next model.

Checks (each returns a pass/fail line)
  schema          required columns exist
  duplicates      one row per (date, product, store)  - a retried job often writes rows twice
  sales_negative  sales must be >= 0
  null_rates      sales <= 5% null, price <= 1% null, forecast_temp <= 10% null
  price_range     0 < price <= 5000 INR
  price_unit      median price per product within 0.5x..2x of the reference (catches INR -> paise, x100)
  unknown_ids     product/store ids must exist in the reference data
  calendar_gaps   no series may miss more than 5% of the days between its first and last date

Tiny example: reference median price of P04 (milk) = 56 INR.  A feed that sends 5,600 has ratio 100 -> FAIL price_unit.
"""
from __future__ import annotations

import pandas as pd

from src.common.paths import REFERENCE_PATH
from src.features.definitions import KEYS

REQUIRED = ["product_id", "store_id", "price", "discount", "promotion", "holiday", "forecast_temp"]


def validate(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
    """Return {'ok': bool, 'problems': [str], 'checks': [(name, passed, detail)]}."""
    if reference is None and REFERENCE_PATH.exists():
        reference = pd.read_csv(REFERENCE_PATH)
    d = df.rename(columns={"forecast_date": "date"})
    checks: list[tuple[str, bool, str]] = []

    miss = [c for c in REQUIRED + ["date"] if c not in d.columns]
    checks.append(("schema", not miss, f"missing columns: {miss}" if miss else "all required columns present"))
    if miss:
        return _finish(checks)

    dup = int(d.duplicated(["date", *KEYS]).sum())
    checks.append(("duplicates", dup == 0, f"{dup} duplicated (date, product, store) rows"))

    if "sales" in d.columns:
        neg = int((d["sales"] < 0).sum())
        checks.append(("sales_negative", neg == 0, f"{neg} rows with negative sales"))
        nr = float(d["sales"].isna().mean())
        checks.append(("null_rate_sales", nr <= 0.05, f"sales null rate {nr:.1%} (limit 5%)"))
    pr = float(d["price"].isna().mean())
    checks.append(("null_rate_price", pr <= 0.01, f"price null rate {pr:.1%} (limit 1%)"))
    tr = float(d["forecast_temp"].isna().mean())
    checks.append(("null_rate_forecast_temp", tr <= 0.10, f"forecast_temp null rate {tr:.1%} (limit 10%)"))

    bad_range = int(((d["price"] <= 0) | (d["price"] > 5000)).sum())
    checks.append(("price_range", bad_range == 0, f"{bad_range} rows outside 0 < price <= 5000"))

    if reference is not None and len(reference):
        ref_med = reference.groupby("product_id")["price"].median()
        cur_med = d.groupby("product_id")["price"].median()
        ratio = (cur_med / ref_med.reindex(cur_med.index)).dropna()
        off = ratio[(ratio < 0.5) | (ratio > 2.0)]
        checks.append(("price_unit", off.empty,
                       "median price ratio vs reference is 0.5-2.0 for every product" if off.empty
                       else "median price ratio out of range: " + ", ".join(f"{k}={v:.0f}x" for k, v in off.items())))
        unknown = sorted((set(d["product_id"]) - set(reference["product_id"])) | (set(d["store_id"]) - set(reference["store_id"])))
        checks.append(("unknown_ids", not unknown, f"unknown ids: {unknown}" if unknown else "all ids known"))

    dd = d.assign(date=pd.to_datetime(d["date"]))
    worst = 0.0
    for _, g in dd.groupby(KEYS):
        span = (g["date"].max() - g["date"].min()).days + 1
        worst = max(worst, 1 - g["date"].nunique() / span)
    checks.append(("calendar_gaps", worst <= 0.05, f"largest share of missing days in any series: {worst:.1%} (limit 5%)"))
    return _finish(checks)


def _finish(checks):
    problems = [f"{n}: {detail}" for n, ok, detail in checks if not ok]
    return dict(ok=not problems, problems=problems, checks=checks)


def print_report(res: dict) -> None:
    for name, ok, detail in res["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26} {detail}")
    print("  DATA QUALITY:", "OK" if res["ok"] else "FAILED")
