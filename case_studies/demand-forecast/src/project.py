"""PROJECT CONTRACT - the only place the shared API / monitoring code looks for project specifics.

Project 3: DEMAND FORECASTING (retail).  Forecast units sold of one product in one store on one future day.

How a real-time request becomes a prediction
--------------------------------------------
    request  (product, store, forecast_date, planned price, discount, promotion, holiday, weather FORECAST)
       |                          the caller does NOT know about lags - the service owns that knowledge
       v
    HistoryStore.features_for()   adds lag_1, lag_7, lag_14, rolling means from the history of THIS series
       |                          (same compute_features() as training  ->  no training-serving skew)
       v
    sklearn Pipeline (one-hot, log, scaling, model)  ->  predicted_units
       v
    interval from validation residual quantiles + demand band vs the recent average

Required names (see template):  TITLE, MODEL_NAME, TARGET, DRIFT_FEATURES, INPUT_COLUMNS, PredictRequest,
PredictResponse, request_to_frame, risk_level, build_response.
Optional hooks used by the shared app: score(), startup(), health_extra(), SCORE_BUCKETS.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits

from src.common.paths import HISTORY_PATH
from src.features.definitions import ALL_CANDIDATE_FEATURES
from src.features.history_store import HistoryStore

TITLE = "Demand Forecast Service"
MODEL_NAME = "demand_forecast_model"
TARGET = "sales"
# logged with every prediction (model input frame) AND present in data/processed/reference.csv
DRIFT_FEATURES = ["lag_1", "rolling_mean_7", "rolling_mean_30", "price", "discount_pct", "forecast_temp"]
INPUT_COLUMNS = ALL_CANDIDATE_FEATURES            # every engineered column the service can build; the artifact picks its own subset
SCORE_BUCKETS = (10, 25, 50, 100, 150, 200, 300, 500, 800, 1500)   # histogram buckets for predicted units


class PredictRequest(BaseModel):
    product_id: str = Field(..., min_length=2, max_length=12, description="e.g. P04")
    store_id: str = Field(..., min_length=2, max_length=12, description="e.g. S2")
    forecast_date: date = Field(..., description="The day to forecast (ISO date). Only history BEFORE this day is used.")
    price: float = Field(..., gt=0, le=5000, description="Planned shelf price in INR per unit (after discount)")
    discount: float = Field(0.0, ge=0, le=5000, description="Discount in INR per unit already inside `price`")
    promotion: int = Field(0, ge=0, le=1, description="1 if a promotion campaign runs that day")
    holiday: int = Field(0, ge=0, le=1, description="1 if the day is a retail holiday")
    forecast_temp: Optional[float] = Field(None, ge=-10, le=55, description="Weather FORECAST in deg C (may be missing)")


class PredictResponse(BaseModel):
    predicted_units: float
    lower: float
    upper: float
    demand_band: str
    model_version: str
    history_last_date: str
    history_stale_days: int
    warnings: list[str] = []


# ----------------------------------------------------------------------------- history store (online features)
_STATE: dict = {"store": None, "error": None, "path": None}


def history_path() -> Path:
    return Path(os.getenv("HISTORY_PATH", HISTORY_PATH))


def startup() -> None:
    """Load the history store once at service start.  Problems are kept (not raised) so /health can report them."""
    path = history_path()
    _STATE.update(store=None, error=None, path=str(path))
    try:
        _STATE["store"] = HistoryStore.from_csv(path)
    except Exception as exc:  # noqa: BLE001  - a broken history must make the service 'degraded', not crash-loop
        _STATE["error"] = f"{type(exc).__name__}: {exc}"


def get_store() -> HistoryStore:
    if _STATE["store"] is None and _STATE["error"] is None:
        startup()
    if _STATE["store"] is None:
        raise RuntimeError(f"history store unavailable: {_STATE['error']}")
    return _STATE["store"]


def health_extra() -> dict:
    s = _STATE["store"]
    return dict(history_loaded=s is not None, history_last_date=str(s.last_date.date()) if s else None,
                history_error=_STATE["error"])


# ----------------------------------------------------------------------------- contract functions
def request_to_frame(req: PredictRequest) -> pd.DataFrame:
    """Request -> one-row frame with ALL features (lags come from the history store)."""
    store = get_store()
    return store.features_for(req.product_id, req.store_id, req.forecast_date, req.price, req.discount,
                              req.promotion, req.holiday, req.forecast_temp)


def score(art: dict, frame: pd.DataFrame) -> float:
    """Point forecast in units (never negative).
    threadpool_limits(1): scoring ONE row gains nothing from multi-threading, and on a busy / small machine
    the thread start-up cost makes a 5 ms call take 300 ms."""
    with threadpool_limits(limits=1):
        return float(max(0.0, art["pipeline"].predict(frame[art["input_columns"]])[0]))


def risk_level(pred: float, frame: pd.DataFrame | None = None) -> str:
    """Demand band versus the recent average (rolling_mean_7):  HIGH >= +20%, LOW <= -20%, else NORMAL.
    Example: recent average 100/day, forecast 130 -> HIGH (stock up); forecast 85 -> NORMAL; forecast 70 -> LOW."""
    if frame is None:
        return "NORMAL"
    base = float(frame["rolling_mean_7"].iloc[0])
    if not np.isfinite(base) or base <= 0:
        return "NORMAL"
    ratio = pred / base
    return "HIGH" if ratio >= 1.2 else "LOW" if ratio <= 0.8 else "NORMAL"


def build_response(req: PredictRequest, p: float, level: str, art: dict) -> dict:
    q = art.get("extra", {}).get("interval", {"low": -0.2, "high": 0.2})
    store = get_store()
    stale = store.staleness_days(req.product_id, req.store_id, req.forecast_date)
    warns = []
    if stale > 0:
        warns.append(f"history is {stale} day(s) stale: lags come from older sales, forecast quality is reduced")
    if req.forecast_temp is None:
        warns.append("forecast_temp missing: median temperature used")
    return dict(predicted_units=round(p, 1), lower=round(max(0.0, p * (1 + q["low"])), 1),
                upper=round(p * (1 + q["high"]), 1), demand_band=level, model_version=art["version"],
                history_last_date=str(store.last_date_for(req.product_id, req.store_id).date()),
                history_stale_days=stale, warnings=warns)
