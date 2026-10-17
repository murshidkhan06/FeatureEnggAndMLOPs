"""Reason codes (a.k.a. adverse-action reasons): WHY did this applicant get this score?

WHY: lenders usually have to tell a declined or referred applicant the main reasons (rules differ by country), and the
underwriter reviewing a MEDIUM case needs them to decide.  A logistic regression makes this easy because the score is a SUM:

    log-odds of default = intercept + sum over features of ( coefficient_j  x  standardized_value_j )

    standardized_value = (value - training mean) / training std       -> "how unusual is this applicant on feature j?"
    contribution_j     = coefficient_j x standardized_value_j         -> push of feature j on the log-odds, relative to the average applicant

A POSITIVE contribution pushes towards default (adverse factor), a NEGATIVE one pushes away.  The top-3 largest positive
contributions are the reason codes.

Tiny example: coefficient(debt_to_income_ratio) = +0.60 per std; an applicant whose DTI is 2 std above average
contributes 0.60 x 2 = +1.20 (the biggest adverse factor).  Another with credit history 1 std ABOVE average and coefficient -0.30
contributes -0.30 (a positive factor, never shown as a reason).

For one-hot categories the contributions of the category columns are summed back to the original feature.
Note: "association, not causation" - a reason code describes how the MODEL scored the applicant, it does not prove that
changing that feature would change the real default risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.pipeline import encoded_to_feature

# Human-readable text per feature (shown to underwriters / applicants).  {v} = the applicant's value where useful.
REASON_TEXT = {
    "debt_to_income_ratio": "Monthly debt payments are high relative to income (DTI {v:.0%})",
    "debt_service_ratio": "Total debt burden after this loan is high relative to monthly income ({v:.0%})",
    "emi_to_income": "The new loan instalment is a large share of monthly income ({v:.0%})",
    "loan_to_income": "The loan amount is large relative to annual income ({v:.2f}x)",
    "log_income": "Declared annual income is low compared with typical applicants",
    "bureau_score": "Credit bureau score is low ({v:.0f})",
    "credit_history_years": "Credit history is short ({v:.1f} years)",
    "employment_years": "Time in current employment is short ({v:.1f} years)",
    "existing_loans": "Number of existing loans is high ({v:.0f})",
    "employment_type": "Employment type ({v}) has historically higher default rates",
    "employment_stability": "Time in current employment is short ({v})",
    "credit_history_bucket": "Credit history is short ({v})",
    "number_of_dependents": "Number of dependents is high ({v:.0f})",
    "loan_term_months": "Long loan term ({v:.0f} months)",
    "loan_amount": "Large loan amount",
    "education": "Education level",
    "city_tier": "City tier ({v})",
    "income_missing": "Annual income was not provided",
    "bureau_score_missing": "No credit bureau score is available (thin file)",
}


BUCKET_WORDS = {"lt_1y": "under 1 year", "1_3y": "1-3 years", "3_7y": "3-7 years", "7y_plus": "7+ years",
                "thin": "thin file", "young": "under 3 years", "established": "3-8 years", "long": "8+ years"}


def contribution_matrix(pipeline, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rows = applicants, columns = original features, values = coefficient x standardized value (log-odds units)."""
    prep, scaler, clf = pipeline.named_steps["prep"], pipeline.named_steps["scale"], pipeline.named_steps["clf"]
    Z = scaler.transform(prep.transform(frame[features]))
    enc_contrib = Z * clf.coef_[0]
    owners = [encoded_to_feature(n, features) for n in prep.get_feature_names_out()]
    out = pd.DataFrame(enc_contrib, columns=owners, index=frame.index)
    return out.T.groupby(level=0, sort=False).sum().T


def reason_codes(pipeline, frame: pd.DataFrame, features: list[str], top_k: int = 3) -> list[list[dict]]:
    """Top-k adverse factors (positive contributions only) for every row of `frame`."""
    contrib = contribution_matrix(pipeline, frame, features)
    result = []
    for i in range(len(frame)):
        row = contrib.iloc[i].sort_values(ascending=False)
        codes = []
        for feat, c in row.items():
            if c <= 0 or len(codes) == top_k:
                break
            val = BUCKET_WORDS.get(frame[feat].iloc[i], frame[feat].iloc[i])
            try:
                text = REASON_TEXT.get(feat, feat).format(v=val)
            except (ValueError, TypeError):
                text = REASON_TEXT.get(feat, feat).split("(")[0].strip()
            if feat == "log_income" and pd.isna(val):
                text = "Annual income was not provided (typical income assumed)"
            codes.append(dict(feature=feat, reason=text, contribution=round(float(c), 3)))
        result.append(codes)
    return result


def standardized_coefficients(pipeline, features: list[str]) -> pd.DataFrame:
    """One row per ORIGINAL feature: sum of |coef| over its encoded columns (all columns are standardized -> comparable)."""
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    t = pd.DataFrame(dict(encoded=names, coef=clf.coef_[0]))
    t["feature"] = [encoded_to_feature(n, features) for n in names]
    t["abs_coef"] = t.coef.abs()
    return t.sort_values("abs_coef", ascending=False).reset_index(drop=True)
