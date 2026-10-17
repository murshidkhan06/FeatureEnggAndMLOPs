"""PROJECT CONTRACT for the credit-scoring service (the only file the shared API / monitoring code reads).

Business: decide at APPLICATION time whether to approve an unsecured personal loan.
Model   : probability of default within 12 months (PD).  Human underwriters review the MEDIUM band.

Prediction-time rule: the request contains ONLY information that exists when the application is submitted.
Sensitive attributes (gender, marital status) and the restricted attribute (age) are NOT part of the request at all:
the scoring service cannot use what it never receives.  `extra="forbid"` also rejects any unexpected field, so a
sensitive or leaky column cannot sneak in through an upstream schema change.
"""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.features.build_features import engineer
from src.features.definitions import FRAME_COLUMNS, LOCATIONS  # noqa: F401  (LOCATIONS re-exported for monitoring)
from src.models.reason_codes import reason_codes

TITLE = "Credit Scoring Service (probability of default)"
MODEL_NAME = "credit_scoring_pd_model"
TARGET = "default_12m"
# Features whose distribution we compare with training data (KS test).  All exist in the model-input frame AND reference.csv.
DRIFT_FEATURES = ["monthly_income", "log_income", "loan_to_income", "debt_to_income_ratio", "emi_to_income",
                  "credit_history_years", "bureau_score"]
INPUT_COLUMNS = FRAME_COLUMNS          # columns request_to_frame() produces; the model uses the SELECTED subset (see artifact)

# Decision bands (probability of default).  Keep in sync with configs/params.yaml (a test checks it).
REFER_CUTOFF = 0.10        # PD >= 10% -> MEDIUM -> human underwriter
DECLINE_CUTOFF = 0.22      # PD >= 22% -> HIGH   -> decline
DECISION_HINT = {"LOW": "APPROVE", "MEDIUM": "REFER_TO_UNDERWRITER", "HIGH": "DECLINE"}


class PredictRequest(BaseModel):
    """One loan application (amounts in INR).  Ranges are a DATA CONTRACT: out-of-range -> HTTP 422."""
    model_config = ConfigDict(extra="forbid")
    annual_income: Optional[float] = Field(None, ge=0, le=100_000_000, description="Declared annual income; may be omitted (about 12% of applicants)")
    monthly_income: float = Field(..., ge=1_000, le=5_000_000, description="Average monthly credits in the last 3 months")
    employment_years: float = Field(..., ge=0, le=50, description="Years in current job/business")
    loan_amount: float = Field(..., ge=10_000, le=5_000_000, description="Requested loan amount")
    loan_term_months: int = Field(..., ge=6, le=84, description="Requested tenure in months")
    credit_history_years: float = Field(..., ge=0, le=50, description="Years since first credit account")
    bureau_score: Optional[float] = Field(None, ge=300, le=900, description="Bureau score; omit for thin-file applicants")
    existing_loans: int = Field(..., ge=0, le=20, description="Open loans at other lenders")
    monthly_debt: float = Field(..., ge=0, le=5_000_000, description="Total monthly EMIs on existing loans")
    number_of_dependents: int = Field(..., ge=0, le=10, description="Financial dependents")
    employment_type: Literal["salaried", "self_employed", "gig", "business"]
    education: Literal["school", "diploma", "graduate", "postgraduate"]
    location: Literal[tuple(LOCATIONS)]                                   # type: ignore[valid-type]


class ReasonCode(BaseModel):
    feature: str
    reason: str
    contribution: float = Field(..., description="coefficient x standardized value (log-odds units); positive = pushes towards default")


class PredictResponse(BaseModel):
    probability_of_default: float
    risk_band: str
    decision_hint: str
    reason_codes: list[ReasonCode]
    model_version: str


def request_to_frame(req: PredictRequest) -> pd.DataFrame:
    """Raw request -> model input row.  Uses the SAME engineer() as training, so no training-serving skew.
    A missing annual_income (None) becomes NaN, exactly like a missing value in the training CSV."""
    raw = pd.DataFrame([req.model_dump()])
    return engineer(raw)[FRAME_COLUMNS]


def risk_level(p: float) -> str:
    return "HIGH" if p >= DECLINE_CUTOFF else "MEDIUM" if p >= REFER_CUTOFF else "LOW"


def build_response(req, p: float, level: str, art: dict) -> dict:
    frame = request_to_frame(req)
    codes = reason_codes(art["pipeline"], frame, art["input_columns"], top_k=3)[0]
    return {"probability_of_default": round(p, 4), "risk_band": level, "decision_hint": DECISION_HINT[level],
            "reason_codes": codes, "model_version": art["version"]}
