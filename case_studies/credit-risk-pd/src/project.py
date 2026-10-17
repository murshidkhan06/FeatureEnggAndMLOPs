"""PROJECT CONTRACT for the credit-risk PD (probability of default) early-warning service.

The shared API (src/api/app.py) and monitoring code only talk to this file.

WHAT THE API RECEIVES
  The engineered feature vector of ONE customer, exactly as an online feature-store lookup would supply it, plus
  customer_id and `features_as_of` (the as_of_date the vector was computed for).  The API does NOT recompute features
  from raw transactions; the monthly batch pipeline (src/features/build_features.py) does that.  What the API DOES
  compute is the 3 derived features, using the very same add_derived_features() function as training -> no skew.

DATA CONTRACT (pydantic Field(ge=, le=)): a value outside the allowed range is rejected with HTTP 422 before it reaches the model.
  Optional fields (credit_utilization, income_stability) may be null: customers without a credit card, or with fewer than
  2 salary credits.  The model pipeline imputes them exactly as it did in training.

PRIVACY: customer_id is echoed in the response but is NOT written to the prediction log (the log holds only model inputs).
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.features.build_features import add_derived_features
from src.features.definitions import BASE_FEATURES, MODEL_FEATURES

TITLE = "Credit Risk PD Early-Warning Service"
MODEL_NAME = "credit_risk_pd_model"
TARGET = "default_next_6m"
DRIFT_FEATURES = list(BASE_FEATURES)            # every raw input is monitored (they are all in the model input frame + reference.csv)
INPUT_COLUMNS = list(MODEL_FEATURES)            # columns of the frame handed to the model pipeline

HIGH_PD = 0.12      # risk bands on the 6-month probability of default
MEDIUM_PD = 0.05


class PredictRequest(BaseModel):
    customer_id: int = Field(..., ge=1, description="Bank customer id (echoed back, never logged)")
    features_as_of: dt.date = Field(..., description="Month-end the features were computed for, e.g. 2023-03-31")
    average_balance: float = Field(..., ge=-5_000_000, le=100_000_000, description="Mean balance over the last 90 days (INR)")
    balance_trend: float = Field(..., ge=-10_000_000, le=10_000_000, description="Slope of 6 month-end balances (INR per month)")
    salary_credit_frequency: float = Field(..., ge=0, le=2, description="Salary credits per month over 6 months")
    loan_payment_ratio: float = Field(..., ge=0, le=1, description="Loan payments made / due over 6 months")
    missed_payment_count: int = Field(..., ge=0, le=6, description="Missed instalments in the last 6 published months")
    credit_utilization: Optional[float] = Field(None, ge=0, le=3, description="Mean card spend / limit; null if no card")
    recent_transaction_count: int = Field(..., ge=0, le=1000, description="Transactions in the last 30 days")
    income_stability: Optional[float] = Field(None, ge=0, le=20, description="Std/mean of salary amounts; null if < 2 salary credits")
    account_age: int = Field(..., ge=0, le=1200, description="Months since signup")


class PredictResponse(BaseModel):
    customer_id: int
    probability_of_default_6m: float
    risk_band: str
    contact_recommended: bool
    model_version: str
    features_as_of: dt.date


def request_to_frame(req: PredictRequest) -> pd.DataFrame:
    """Request -> one-row model input frame.  Derived features come from the SAME function as in training."""
    row = {f: getattr(req, f) for f in BASE_FEATURES}
    frame = pd.DataFrame([row]).astype(float)           # None -> NaN
    return add_derived_features(frame)[INPUT_COLUMNS]


def risk_level(p: float) -> str:
    """HIGH >= 12 %, MEDIUM >= 5 %, otherwise LOW.   Example: p = 0.07 -> MEDIUM."""
    return "HIGH" if p >= HIGH_PD else "MEDIUM" if p >= MEDIUM_PD else "LOW"


def build_response(req: PredictRequest, p: float, level: str, art: dict) -> dict:
    return {"customer_id": req.customer_id, "probability_of_default_6m": round(p, 4), "risk_band": level,
            "contact_recommended": bool(p >= art["threshold"]), "model_version": art["version"],
            "features_as_of": req.features_as_of}
