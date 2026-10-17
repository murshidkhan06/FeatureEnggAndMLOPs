"""Data-quality validation of an incoming batch - the FIRST rung of the retraining decision ladder.

WHY first?  If the data is broken (a unit changed, a column was renamed, nulls exploded) the model's scores are garbage.
The fix is in the PIPELINE, not a new model.  Retraining on broken data would teach the model the bug.

Checks (each returns a problem string; no problems = OK)
  schema        all request columns present
  ranges        values inside the API contract (the same ge/le limits as PredictRequest)
  null rates    annual_income may be missing (~12% is normal); everything else must be complete
  categories    employment_type / education / location are known values
  unit sanity   monthly_income should be about annual_income / 12.  Tiny example: annual 600,000 -> monthly ~50,000, ratio ~1.0.
                If the median ratio jumps to 12 (or 0.083) somebody changed the unit of one field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import project as P
from src.data.make_dataset import REQUEST_FIELDS

EXPECTED_ANNUAL_INCOME_NULL = (0.03, 0.25)    # normal band for the share of missing annual_income


def _limits(name: str) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in P.PredictRequest.model_fields[name].metadata:
        lo = getattr(m, "ge", lo)
        hi = getattr(m, "le", hi)
    return lo, hi


def validate_batch(df: pd.DataFrame) -> dict:
    problems: list[str] = []
    checks: dict[str, str] = {}
    missing = [c for c in REQUEST_FIELDS if c not in df.columns]
    if missing:
        problems.append(f"schema: missing columns {missing}")
        checks["schema"] = "FAIL"
        return dict(ok=False, problems=problems, checks=checks)
    checks["schema"] = "OK"

    bad_range = []
    for c in REQUEST_FIELDS:
        if c in ("employment_type", "education", "location"):
            continue
        lo, hi = _limits(c)
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        share = float(((s < lo) | (s > hi)).mean()) if len(s) else 0.0
        if share > 0.01:
            bad_range.append(f"{c}: {share:.1%} outside [{lo}, {hi}]")
    checks["ranges"] = "OK" if not bad_range else "FAIL"
    problems += [f"ranges: {b}" for b in bad_range]

    bad_null = []
    for c in REQUEST_FIELDS:
        rate = float(df[c].isna().mean())
        if c == "annual_income":
            lo, hi = EXPECTED_ANNUAL_INCOME_NULL
            if not lo <= rate <= hi:
                bad_null.append(f"annual_income: {rate:.1%} missing (normal {lo:.0%}-{hi:.0%})")
        elif c == "bureau_score":
            if rate > 0.20:
                bad_null.append(f"bureau_score: {rate:.1%} missing (normal ~8%)")
        elif rate > 0.01:
            bad_null.append(f"{c}: {rate:.1%} missing (must be complete)")
    checks["null_rates"] = "OK" if not bad_null else "FAIL"
    problems += [f"nulls: {b}" for b in bad_null]

    bad_cat = []
    for c, allowed in (("employment_type", ["salaried", "self_employed", "gig", "business"]),
                       ("education", ["school", "diploma", "graduate", "postgraduate"]), ("location", P.LOCATIONS)):
        share = float((~df[c].isin(allowed)).mean())
        if share > 0.005:
            bad_cat.append(f"{c}: {share:.1%} unknown values")
    checks["categories"] = "OK" if not bad_cat else "FAIL"
    problems += [f"categories: {b}" for b in bad_cat]

    both = df[["annual_income", "monthly_income"]].dropna()
    both = both[both.annual_income > 0]
    ratio = float(np.median(both.monthly_income / (both.annual_income / 12))) if len(both) else float("nan")
    if not 0.7 <= ratio <= 1.4:
        problems.append(f"unit sanity: median(monthly_income / (annual_income/12)) = {ratio:.2f}, expected ~1.0 -> a unit or field mapping changed upstream")
        checks["unit_sanity"] = "FAIL"
    else:
        checks["unit_sanity"] = "OK"
    return dict(ok=not problems, problems=problems, checks=checks, income_unit_ratio=round(ratio, 3))
