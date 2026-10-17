"""Single source of truth for HOW each column is treated (type, order, text definition).

WHY: preprocessing, feature selection, the API and the documentation must all agree on what a column
IS (numeric? ordered category? nominal category?).  Disagreement here is a classic cause of
training-serving skew.  Governance metadata (owner, approval status, availability at prediction time)
lives separately in configs/feature_registry.yaml - a test checks that both files list the same names.

kind
  numeric   -> median imputation (+ scaling)
  binary    -> 0/1 flag (a missing-value indicator, for example)
  ordinal   -> ordered categories -> integer codes 0,1,2,... (order matters)
  nominal   -> unordered categories -> one-hot encoding (one 0/1 column per category)
"""
from __future__ import annotations

from src.data.make_dataset import CITIES

ASSUMED_ANNUAL_RATE_PCT = 14.0       # ASSUMPTION for emi_estimate (documented in docs/DATASET.md)
LOCATIONS = list(CITIES)             # 15 cities
CITY_TIER = {c: f"tier{v[0]}" for c, v in CITIES.items()}

EDUCATION_ORDER = ["school", "diploma", "graduate", "postgraduate"]
STABILITY_ORDER = ["lt_1y", "1_3y", "3_7y", "7y_plus"]            # employment_stability buckets
HISTORY_ORDER = ["thin", "young", "established", "long"]          # credit_history_bucket buckets

# name -> (kind, categories or None)
FEATURE_KINDS: dict[str, tuple[str, list[str] | None]] = {
    # ---- raw columns known when the application is submitted ----
    "annual_income": ("numeric", None), "monthly_income": ("numeric", None),
    "employment_years": ("numeric", None), "loan_amount": ("numeric", None),
    "loan_term_months": ("numeric", None), "credit_history_years": ("numeric", None),
    "bureau_score": ("numeric", None), "existing_loans": ("numeric", None),
    "monthly_debt": ("numeric", None), "debt_to_income_ratio": ("numeric", None),
    "number_of_dependents": ("numeric", None),
    "employment_type": ("nominal", ["salaried", "self_employed", "gig", "business"]),
    "education": ("ordinal", EDUCATION_ORDER),
    "location": ("nominal", LOCATIONS),
    # ---- engineered from the raw columns above (see build_features.engineer) ----
    "log_income": ("numeric", None), "loan_to_income": ("numeric", None), "emi_estimate": ("numeric", None),
    "emi_to_income": ("numeric", None), "debt_service_ratio": ("numeric", None),
    "income_missing": ("binary", None), "bureau_score_missing": ("binary", None),
    "employment_stability": ("ordinal", STABILITY_ORDER), "credit_history_bucket": ("ordinal", HISTORY_ORDER),
    "city_tier": ("nominal", ["tier1", "tier2", "tier3"]),
    # ---- columns that exist in the raw data but must NOT be model inputs (kept for audits / demos) ----
    "age": ("numeric", None), "gender": ("nominal", ["female", "male"]),
    "marital_status": ("nominal", ["single", "married", "divorced"]),
    "interest_rate_offered": ("numeric", None), "loan_approved": ("binary", None),
    "days_past_due_after_disbursal": ("numeric", None),
    "collections_status": ("nominal", ["none", "reminder", "legal", "written_off"]),
    "post_disbursal_payment_ratio": ("numeric", None),
    "credit_officer_override": ("nominal", ["none", "upgrade", "downgrade"]),
}

ENGINEERED = ["log_income", "loan_to_income", "emi_estimate", "emi_to_income", "debt_service_ratio",
              "income_missing", "bureau_score_missing", "employment_stability", "credit_history_bucket", "city_tier"]
RAW_ELIGIBLE = ["annual_income", "monthly_income", "employment_years", "loan_amount", "loan_term_months",
                "credit_history_years", "bureau_score", "existing_loans", "monthly_debt", "debt_to_income_ratio",
                "number_of_dependents", "employment_type", "education", "location"]
SENSITIVE = ["gender", "marital_status"]
GOVERNANCE_FLAGGED = ["age", "location"]
LEAKAGE = ["interest_rate_offered", "loan_approved", "days_past_due_after_disbursal", "collections_status",
           "post_disbursal_payment_ratio", "credit_officer_override"]

# Business-sense check: for each single-column feature, does a HIGHER value mean HIGHER default risk (+1) or LOWER (-1)?
# A fitted coefficient with the opposite sign is a red flag (usually multicollinearity) and cannot be explained to a
# regulator or an applicant ("why does a bigger instalment make me safer?").
EXPECTED_SIGN = {
    "annual_income": -1, "monthly_income": -1, "log_income": -1, "bureau_score": -1, "credit_history_years": -1,
    "employment_years": -1, "employment_stability": -1, "credit_history_bucket": -1, "education": -1,
    "debt_to_income_ratio": +1, "loan_to_income": +1, "emi_to_income": +1, "debt_service_ratio": +1, "existing_loans": +1,
    "number_of_dependents": +1, "loan_amount": +1, "loan_term_months": +1, "emi_estimate": +1, "monthly_debt": +1,
    "bureau_score_missing": +1,
}

# Columns produced by engineer() for a REQUEST (what the scoring service can build at application time).
FRAME_COLUMNS = RAW_ELIGIBLE + ENGINEERED

# Plain-language definition + formula + one worked example (used by docs, notebook and results.json)
DEFINITIONS = {
    "log_income": dict(definition="Natural log of (1 + annual income): tames the right-skewed income distribution.",
                       formula="log(1 + annual_income)", example_in={"annual_income": 600000}, example_out=13.304687),
    "loan_to_income": dict(definition="How many years of income the loan equals.",
                           formula="loan_amount / annual_income", example_in={"loan_amount": 300000, "annual_income": 600000},
                           example_out=0.5),
    "emi_estimate": dict(definition="Monthly instalment if the loan were charged the ASSUMED rate (14% p.a.).",
                         formula="EMI = P*r*(1+r)^n / ((1+r)^n - 1),  r = annual_rate/12",
                         example_in={"loan_amount": 300000, "loan_term_months": 24}, example_out=14404.0),
    "emi_to_income": dict(definition="Share of monthly income the NEW loan instalment would take.",
                          formula="emi_estimate / monthly_income", example_in={"emi_estimate": 14404, "monthly_income": 50000},
                          example_out=0.2881),
    "debt_service_ratio": dict(definition="Share of monthly income going to ALL debt payments after this loan.",
                               formula="(monthly_debt + emi_estimate) / monthly_income",
                               example_in={"monthly_debt": 10000, "emi_estimate": 14404, "monthly_income": 50000}, example_out=0.4881),
    "income_missing": dict(definition="1 if the applicant did not report annual income (missingness can itself be informative).",
                           formula="1 if annual_income is missing else 0", example_in={"annual_income": None}, example_out=1),
    "bureau_score_missing": dict(definition="1 if there is no bureau score (thin-file applicant).",
                                 formula="1 if bureau_score is missing else 0", example_in={"bureau_score": None}, example_out=1),
    "employment_stability": dict(definition="Bucket of employment_years: lt_1y / 1_3y / 3_7y / 7y_plus.",
                                 formula="cut(employment_years, [0,1,3,7,inf))", example_in={"employment_years": 2.0}, example_out="1_3y"),
    "credit_history_bucket": dict(definition="Bucket of credit_history_years: thin / young / established / long.",
                                  formula="cut(credit_history_years, [0,1,3,8,inf))", example_in={"credit_history_years": 5.0},
                                  example_out="established"),
    "city_tier": dict(definition="Coarse city class (tier1/2/3) instead of the fine-grained city (proxy-risk control).",
                      formula="lookup(location)", example_in={"location": "Mumbai"}, example_out="tier1"),
    "debt_to_income_ratio": dict(definition="Monthly debt payments divided by monthly income.",
                                 formula="monthly_debt / monthly_income", example_in={"monthly_debt": 10000, "monthly_income": 50000},
                                 example_out=0.2),
}


def kind_of(name: str) -> str:
    return FEATURE_KINDS[name][0]


def categories_of(name: str) -> list[str] | None:
    return FEATURE_KINDS[name][1]
