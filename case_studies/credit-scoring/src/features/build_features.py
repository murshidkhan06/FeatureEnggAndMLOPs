"""Turn raw application columns into model features - the SAME code for training, batch scoring and the API.

WHY one function?  Training-serving skew (the model sees one version of a feature in training and a different
version in production) is mostly caused by two teams re-implementing the same formula twice.  Here the
`engineer()` function below is imported by the training code AND by `src/project.py` (the API).

Only STATELESS, row-wise formulas live here (they never look at other rows).  Anything that must be LEARNED from
data (median for imputation, mean/std for scaling, category lists...) lives inside the sklearn Pipeline so it
is fitted on the training data only.

Run:  python -m src.features.build_features        -> writes data/processed/{train,valid,test}.csv + reference.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.definitions import ASSUMED_ANNUAL_RATE_PCT, CITY_TIER, HISTORY_ORDER, STABILITY_ORDER


def emi(principal, months, annual_rate_pct: float = ASSUMED_ANNUAL_RATE_PCT):
    """Standard EMI (equated monthly instalment) formula.

    WHY: a loan of 3 lakh over 12 months and the same loan over 60 months are very different burdens for a
    household; the instalment captures that, loan_amount alone does not.

        EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)      P = principal, r = monthly rate, n = months

    Worked example: P = 300,000  n = 24  annual rate 14%  ->  r = 0.14/12 = 0.0116667
        (1+r)^24 = 1.32099        EMI = 300000 * 0.0116667 * 1.32099 / 0.32099 = 14,404 (approx.)
    """
    r = annual_rate_pct / 100.0 / 12.0
    p = np.asarray(principal, dtype=float)
    n = np.asarray(months, dtype=float)
    growth = (1.0 + r) ** n
    return p * r * growth / (growth - 1.0)


def _bucket(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    """Right-open buckets [e0,e1), [e1,e2), ...  Missing values stay missing (they become the string 'nan')."""
    return pd.cut(values, bins=edges, labels=labels, right=False).astype(object).astype(str)


def engineer(df: pd.DataFrame, annual_rate_pct: float = ASSUMED_ANNUAL_RATE_PCT) -> pd.DataFrame:
    """Add the engineered columns to a frame holding the raw application fields.

    Input columns needed : annual_income (may be NaN), monthly_income, employment_years, loan_amount,
                           loan_term_months, credit_history_years, bureau_score (may be NaN), monthly_debt, location
    Returns a COPY with these extra columns (formulas in src/features/definitions.py):
        debt_to_income_ratio, income_missing, bureau_score_missing, log_income, loan_to_income, emi_estimate,
        emi_to_income, debt_service_ratio, employment_stability, credit_history_bucket, city_tier

    Worked example (one applicant)
        annual_income=600,000  monthly_income=50,000  monthly_debt=10,000  loan=300,000  term=24 months
        debt_to_income_ratio = 10,000/50,000 = 0.20           loan_to_income = 300,000/600,000 = 0.50
        log_income = log(1+600,000) = 13.3047                 emi_estimate = 14,404 (see emi())
        emi_to_income = 14,404/50,000 = 0.288                 debt_service_ratio = (10,000+14,404)/50,000 = 0.488
    """
    out = df.copy()
    for c in ("annual_income", "bureau_score"):
        if c not in out:
            out[c] = np.nan
    income = pd.to_numeric(out["annual_income"], errors="coerce")
    monthly = pd.to_numeric(out["monthly_income"], errors="coerce").replace(0, np.nan)
    out["annual_income"] = income
    out["debt_to_income_ratio"] = (out["monthly_debt"] / monthly).round(4)
    out["income_missing"] = income.isna().astype(int)
    out["bureau_score"] = pd.to_numeric(out["bureau_score"], errors="coerce")   # None (API) -> NaN, same as training
    out["bureau_score_missing"] = out["bureau_score"].isna().astype(int)
    out["log_income"] = np.log1p(income)                                   # NaN stays NaN: imputed later, in the Pipeline
    out["loan_to_income"] = out["loan_amount"] / income.where(income > 0)
    out["emi_estimate"] = emi(out["loan_amount"], out["loan_term_months"], annual_rate_pct)
    out["emi_to_income"] = out["emi_estimate"] / monthly
    out["debt_service_ratio"] = (out["monthly_debt"] + out["emi_estimate"]) / monthly
    out["employment_stability"] = _bucket(out["employment_years"], [-np.inf, 1, 3, 7, np.inf], STABILITY_ORDER)
    out["credit_history_bucket"] = _bucket(out["credit_history_years"], [-np.inf, 1, 3, 8, np.inf], HISTORY_ORDER)
    out["city_tier"] = out["location"].map(CITY_TIER)
    return out


def time_split(df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15, date_col: str = "application_date"):
    """Split by TIME (oldest -> train, middle -> valid, newest -> test) and remove re-applicants from earlier parts.

    WHY not a random split?  (1) In production we predict the FUTURE from the PAST; a random split lets the model
    peek at the future.  (2) The same person can apply twice; a random split puts one application in train and
    the near-identical other one in test, so the test score is flattered (applicant leakage).

    Rule: rows are sorted by date; the first `train_frac` are train, the next `valid_frac` valid, the rest test.
    Any TRAIN row whose applicant_id also appears in valid/test is dropped, and valid rows whose id appears
    in test are dropped.

    Tiny example: 10 rows, 0.7/0.15 -> rows 0-6 train, 7 valid, 8-9 test.
    """
    d = df.sort_values([date_col, "applicant_id"], kind="stable").reset_index(drop=True)
    n = len(d)
    i1, i2 = int(n * train_frac), int(n * (train_frac + valid_frac))
    train, valid, test = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    valid = valid[~valid["applicant_id"].isin(test["applicant_id"])]
    train = train[~train["applicant_id"].isin(set(valid["applicant_id"]) | set(test["applicant_id"]))]
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    from src.common.params import load_params
    from src.common.paths import DATA_PROCESSED, DATA_RAW, REFERENCE_PATH

    prm = load_params()
    raw = pd.read_csv(DATA_RAW / "applications.csv")
    full = engineer(raw, prm["features"]["assumed_annual_rate_pct"])
    train, valid, test = time_split(full, prm["split"]["train_frac"], prm["split"]["valid_frac"])
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        part.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):6d} rows  {part.application_date.min()} -> {part.application_date.max()}  "
              f"default rate {part['default_12m'].mean():.3f}")
    # reference sample for drift checks = what the model SAW at training time (model-input style columns only)
    from src.features.definitions import FRAME_COLUMNS
    train[FRAME_COLUMNS].sample(3000, random_state=0).to_csv(REFERENCE_PATH, index=False)
    print(f"reference sample (3000 rows of the training feature frame) -> {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
