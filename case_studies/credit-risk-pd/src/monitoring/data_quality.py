"""DATA QUALITY gate: run BEFORE any drift analysis or retraining decision.

WHY first?  Drift tests tell you "the numbers changed".  But a broken upstream job (a migration that leaves a column empty,
a unit change, duplicated rows) also changes the numbers - and the cure is to FIX THE PIPELINE, not to retrain a model on
garbage.  So the decision ladder always starts with data quality.

CHECKS
  schema       all required columns exist
  one snapshot the batch belongs to a single features_as_of date, one row per customer
  ranges       values inside the API data-contract ranges (Field(ge=, le=))
  null rates   null rate of every feature may not exceed the REFERENCE null rate by more than `max_increase` (default 5 percentage points)

TINY EXAMPLE
  salary_credit_frequency: reference null rate 0 %, this month 38 %  -> increase 38 pp > 5 pp -> FAIL  (source-system migration?)
  credit_utilization:      reference null rate 16 %, this month 17 %  -> increase 1 pp -> OK    (no-card customers are normal)
"""
from __future__ import annotations

import pandas as pd

from src import project as P
from src.features.definitions import BASE_FEATURES


def contract_ranges() -> dict[str, tuple[float | None, float | None]]:
    """Allowed (min, max) of every request field, read from the pydantic data contract so there is ONE source of truth."""
    out = {}
    for name, info in P.PredictRequest.model_fields.items():
        lo = hi = None
        for m in info.metadata:
            lo = getattr(m, "ge", lo)
            hi = getattr(m, "le", hi)
        out[name] = (lo, hi)
    return out


def validate_batch(prod: pd.DataFrame, reference: pd.DataFrame, max_increase: float = 0.05) -> dict:
    """Return {'ok': bool, 'problems': [str], 'table': DataFrame(feature, ref_null, prod_null, increase_pp, out_of_range)}."""
    problems: list[str] = []
    missing = [c for c in ["customer_id", "features_as_of"] + BASE_FEATURES if c not in prod.columns]
    if missing:
        return {"ok": False, "problems": [f"missing columns: {missing}"], "table": pd.DataFrame()}
    if prod["features_as_of"].nunique() != 1:
        problems.append(f"batch mixes {prod['features_as_of'].nunique()} different features_as_of dates")
    if prod["customer_id"].duplicated().any():
        problems.append(f"{int(prod['customer_id'].duplicated().sum())} duplicated customer_id rows")
    ranges = contract_ranges()
    rows = []
    for f in BASE_FEATURES:
        ref_null = float(reference[f].isna().mean()) if f in reference.columns else 0.0
        null = float(prod[f].isna().mean())
        lo, hi = ranges[f]
        v = pd.to_numeric(prod[f], errors="coerce")
        oor = int(((lo is not None) & (v < lo)).sum() + ((hi is not None) & (v > hi)).sum())
        rows.append(dict(feature=f, ref_null=round(ref_null, 4), prod_null=round(null, 4),
                         increase_pp=round((null - ref_null) * 100, 1), out_of_range=oor))
        if null - ref_null > max_increase:
            problems.append(f"{f}: null rate {null:.0%} vs {ref_null:.0%} in the reference (+{(null - ref_null) * 100:.0f} pp)")
        if oor:
            problems.append(f"{f}: {oor} values outside the allowed range [{lo}, {hi}]")
    return {"ok": not problems, "problems": problems, "table": pd.DataFrame(rows)}
