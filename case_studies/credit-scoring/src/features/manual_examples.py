"""Hand calculation FIRST, code second: every engineered feature on ONE applicant.

WHY: if you cannot reproduce a feature with a calculator you do not understand it, and nobody can audit it.
This script prints the hand calculation (plain Python arithmetic, step by step) next to what the production
function `engineer()` returns, and checks that they agree.  It also shows the three small ideas behind the
preprocessing pipeline: the log transform, median imputation + missing indicator, and standardization.

The applicant (amounts in INR)
    annual_income 600,000   monthly_income 50,000   monthly_debt 10,000   loan 300,000 over 24 months
    employment 2.0 years    credit history 5.0 years   bureau_score 680   location Mumbai
Assumed interest rate for the EMI estimate: 14% per year (params.yaml features.assumed_annual_rate_pct) - an ASSUMPTION,
the real rate is only set AFTER scoring (interest_rate_offered is a leakage column).

Run:  python -m src.features.manual_examples
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.common.paths import REPORTS
from src.features.build_features import engineer

APPLICANT = dict(annual_income=600_000.0, monthly_income=50_000.0, monthly_debt=10_000.0, loan_amount=300_000.0,
                 loan_term_months=24, employment_years=2.0, credit_history_years=5.0, bureau_score=680.0, location="Mumbai")
RATE = 14.0


def hand_calculation() -> dict:
    """Plain-Python arithmetic: no pandas, no numpy - exactly what you would do on paper."""
    a = APPLICANT
    r = RATE / 100 / 12                                            # monthly interest rate
    growth = (1 + r) ** a["loan_term_months"]                      # (1+r)^n
    emi = a["loan_amount"] * r * growth / (growth - 1)
    steps = dict(r=r, growth=growth, emi=emi)
    feats = {
        "debt_to_income_ratio": a["monthly_debt"] / a["monthly_income"],
        "log_income": math.log(1 + a["annual_income"]),
        "loan_to_income": a["loan_amount"] / a["annual_income"],
        "emi_estimate": emi,
        "emi_to_income": emi / a["monthly_income"],
        "debt_service_ratio": (a["monthly_debt"] + emi) / a["monthly_income"],
        "income_missing": 0,
        "employment_stability": "1_3y",                            # 2.0 years falls in [1, 3)
        "credit_history_bucket": "established",                    # 5.0 years falls in [3, 8)
        "city_tier": "tier1",                                      # Mumbai is a tier-1 city
    }
    return dict(steps=steps, features=feats)


def main() -> None:
    hand = hand_calculation()
    s = hand["steps"]
    print("EMI by hand (the standard loan formula)")
    print(f"   monthly rate r = 14% / 12 / 100          = {s['r']:.6f}")
    print(f"   growth (1 + r)^24                        = {s['growth']:.5f}")
    print(f"   EMI = P * r * growth / (growth - 1)      = 300,000 * {s['r']:.6f} * {s['growth']:.5f} / {s['growth'] - 1:.5f} = {s['emi']:,.2f}")
    print("   (sanity: 24 payments of ~14,404 = ~345,700 -> about 45,700 of interest on a 300,000 loan)\n")

    code = engineer(pd.DataFrame([APPLICANT])).iloc[0]
    rows, ok = [], True
    for name, h in hand["features"].items():
        c = code[name]
        same = (h == c) if isinstance(h, str) else abs(float(h) - float(c)) < 1e-3
        ok &= bool(same)
        rows.append(dict(feature=name, hand_calculation=round(h, 4) if not isinstance(h, str) else h,
                         engineer_code=round(float(c), 4) if not isinstance(c, str) else c, agree="YES" if same else "NO"))
    t = pd.DataFrame(rows)
    print("HAND CALCULATION vs engineer() for one applicant")
    print(t.to_string(index=False))
    print(f"\nall features agree: {ok}\n")
    print("Business reading: this applicant would pay 28.8% of monthly income for the NEW loan alone, and 48.8% for ALL debts together;\n"
          "a debt-service ratio above ~50% is a classic warning sign for lenders.\n")

    # ---- the log transform --------------------------------------------------------------------------------------------
    inc = np.array([300_000, 450_000, 600_000, 900_000, 6_000_000.0])        # the last one is a data-entry error (x10)
    print("LOG TRANSFORM: log(1 + income) pulls an extreme value back towards the others")
    print("   income      :", [f"{v:,.0f}" for v in inc])
    print("   log(1+inc)  :", [round(float(v), 2) for v in np.log1p(inc)])
    print(f"   biggest/median = {inc.max() / np.median(inc):.1f}x on the raw scale, but only {np.log1p(inc).max() - np.median(np.log1p(inc)):.2f} log units above the median.\n")

    # ---- median imputation + missing indicator ------------------------------------------------------------------------
    train_income = np.array([10.0, 20.0, 90.0, np.nan])
    med = float(np.nanmedian(train_income))
    print("MEDIAN IMPUTATION (learned on TRAIN only) + MISSING INDICATOR")
    print(f"   train values {train_income.tolist()} -> median of the observed values = {med}  -> NaN becomes {med}")
    print("   income_missing = 1 for that row: the model can still learn that 'did not report income' is informative.")
    print("   The API does exactly the same: annual_income omitted -> NaN -> engineer() -> same Pipeline -> same median.\n")

    # ---- standardization ---------------------------------------------------------------------------------------------
    x = np.array([2.0, 4.0, 6.0])
    print("STANDARDIZATION z = (x - mean) / std  (mean and std also learned on TRAIN only)")
    print(f"   x = {x.tolist()}  mean = {x.mean()}  std = {x.std():.4f}  ->  z = {np.round((x - x.mean()) / x.std(), 3).tolist()}")
    print("   After this every coefficient reads 'change in log-odds of default per +1 standard deviation' -> comparable and explainable.")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "feature_examples.json").write_text(json.dumps(dict(applicant=APPLICANT, assumed_rate_pct=RATE, emi_steps=s, table=rows), indent=2, default=str))


if __name__ == "__main__":
    main()
