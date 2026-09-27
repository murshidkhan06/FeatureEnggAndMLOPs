"""
api/main.py

FastAPI application -- refactored from the original teaching project's app/main.py, with
the SAME contract (same endpoints, same request/response schema, same "zero feature
engineering logic in the API" design) plus observability instrumentation added on top:
structured logging, per-prediction logging to monitoring/predictions.csv, and a /metrics
endpoint.

Run locally with:
    uvicorn api.main:app --reload --port 8000

Then try:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
        "study_hours": 12.5, "attendance_pct": 88.0,
        "mock_test_1": 72.0, "mock_test_2": 75.0, "mock_test_3": 70.0,
        "income_bracket": "Medium", "city": "Pune",
        "enrollment_date": "2025-06-01", "shoe_size": 9.0, "lucky_number": 42
    }'

Design note (unchanged from the original app/main.py): this file contains ZERO feature
engineering logic of its own. Raw request in -> the saved pipeline -> prediction out. That
is exactly what prevents "the API preprocesses data slightly differently than training
did" -- one of the most common real causes of a model behaving worse in production than in
testing (training-serving skew).
"""
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# IMPORTANT: this import is not decorative. The pickled pipeline contains a custom
# FeatureCreator step, and joblib only stores a REFERENCE to where that class is defined,
# not its code. Without this import, loading the pickle below fails with
# "AttributeError: Can't get attribute 'FeatureCreator'" -- see src/features.py's
# docstring for the full story of this exact bug (caught for real while building the
# original version of this project).
from src.features import FeatureCreator  # noqa: F401
from src.monitoring import log_prediction, logger, metrics, record_actual_label, timed

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "exam_score_pipeline.joblib"
MODEL_VERSION = "unknown"  # set at startup, once the model is actually loaded

# Loaded once, at startup -- NOT on every request. This is the same fitted pipeline object
# produced by `python -m src.train`, nothing reimplemented here.
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, MODEL_VERSION
    if not MODEL_PATH.exists():
        # Fail loudly and clearly at startup, rather than 500ing confusingly on the first
        # request -- PHASE 28's "missing model file" error handling requirement.
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python -m src.train` (or `dvc "
            f"repro`) first to train and save the pipeline."
        )
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}") from e

    MODEL_VERSION = str(MODEL_PATH.stat().st_mtime_ns)  # cheap, always-available version tag
    logger.info(f"Model loaded from {MODEL_PATH} (version={MODEL_VERSION})")
    yield
    pipeline = None  # nothing else to clean up -- included for symmetry/clarity


app = FastAPI(
    title="Student Exam Score Predictor",
    description="Predicts a student's final exam score from raw, unprocessed profile data. "
                "All feature engineering (imputation, scaling, encoding, PCA, feature "
                "selection) happens automatically inside the loaded pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


class StudentInput(BaseModel):
    """Raw, unprocessed student data -- exactly the same raw columns the pipeline was
    trained on. No preprocessed/derived fields belong here; the pipeline computes those
    itself."""

    study_hours: float = Field(..., ge=0, le=40, description="Weekly study hours")
    attendance_pct: float = Field(..., ge=0, le=100, description="Attendance percentage")
    mock_test_1: float = Field(..., ge=0, le=100)
    mock_test_2: float = Field(..., ge=0, le=100)
    mock_test_3: float = Field(..., ge=0, le=100)
    income_bracket: Literal["Low", "Medium", "High"]
    city: str
    enrollment_date: str = Field(..., description="YYYY-MM-DD")
    shoe_size: float = Field(..., description="Deliberately irrelevant -- kept to show the pipeline ignores it")
    lucky_number: int = Field(..., description="Deliberately irrelevant -- kept to show the pipeline ignores it")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "study_hours": 12.5,
            "attendance_pct": 88.0,
            "mock_test_1": 72.0,
            "mock_test_2": 75.0,
            "mock_test_3": 70.0,
            "income_bracket": "Medium",
            "city": "Pune",
            "enrollment_date": "2025-06-01",
            "shoe_size": 9.0,
            "lucky_number": 42,
        }
    })


class PredictionOutput(BaseModel):
    predicted_final_score: float
    model_version: str
    request_id: str


class FeedbackInput(BaseModel):
    """A DELAYED LABEL arriving for a past prediction -- see src/monitoring.py's
    `record_actual_label` docstring. In this teaching project a student would submit this
    once their real exam result is known, days or weeks after the original /predict call."""

    request_id: str = Field(..., description="The request_id returned by the original /predict call")
    actual_final_score: float = Field(..., ge=0, le=100)


class FeedbackOutput(BaseModel):
    status: str
    request_id: str


@app.get("/health")
def health():
    """Combined liveness+readiness check, kept for backward compatibility with existing
    tooling/tests. New integrations (e.g. a Kubernetes-style deployment) should prefer the
    separate /health/live and /health/ready endpoints below -- see their docstrings for why
    the two questions are not the same."""
    return {"status": "ok", "model_loaded": pipeline is not None, "model_version": MODEL_VERSION}


@app.get("/health/live")
def health_live():
    """LIVENESS: "is the process itself still running and able to respond at all?" This
    answers yes as long as the FastAPI app is up and this handler can execute -- it does
    NOT check whether the model is loaded. In a Kubernetes-style deployment, a FAILED
    liveness check means "kill and restart this container" -- so it should only ever fail
    for problems a RESTART would actually fix (e.g. the process is deadlocked), never for
    "the model isn't ready yet", which restarting does nothing to help."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    """READINESS: "is this instance currently able to correctly serve real requests?" This
    checks that the model has actually finished loading. In a Kubernetes-style deployment,
    a FAILED readiness check means "stop sending this instance traffic (but don't kill it)"
    -- exactly the right response while a container is still starting up (model loading can
    take a few seconds) or, in a fancier deployment, while a rolling model swap is in
    progress. Liveness answers "is it alive?"; readiness answers "is it ready for work?" --
    a process can be alive but not ready (starting up), but never ready while not alive."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded -- not ready for traffic.")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.post("/feedback", response_model=FeedbackOutput)
def feedback(fb: FeedbackInput):
    """Records the ACTUAL outcome for a past prediction -- the delayed label that makes
    performance-decay monitoring possible (src/performance_monitor.py). This does NOT
    require the original request_id to be re-validated against the model itself; it just
    appends to the prediction store's labels file. Called once the true final_score becomes
    known -- normally much later than the original /predict call, hence "delayed"."""
    record_actual_label(request_id=fb.request_id, actual_final_score=fb.actual_final_score)
    logger.info(f"[{fb.request_id}] recorded actual_final_score={fb.actual_final_score}")
    return FeedbackOutput(status="recorded", request_id=fb.request_id)


@app.get("/metrics")
def get_metrics():
    """Basic application metrics -- request/prediction/error counts and average latency.
    Deliberately a plain JSON endpoint rather than requiring Prometheus, per PHASE 15's
    "keep the first implementation simple and local" instruction. A Prometheus-format
    /metrics endpoint is a natural upgrade path, noted in the README."""
    return metrics.snapshot()


@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    """Predict a student's final exam score from raw input data.

    The pipeline handles imputation, scaling, encoding, PCA, and feature selection
    internally -- this endpoint just validates the request shape, calls .predict(), and
    logs the outcome for observability.
    """
    metrics.record_request()
    request_id = str(uuid.uuid4())

    if pipeline is None:
        metrics.record_error()
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_df = pd.DataFrame([student.model_dump()])

    with timed() as elapsed:
        try:
            prediction = float(pipeline.predict(input_df)[0])
        except Exception as e:
            latency_ms = elapsed()
            metrics.record_error()
            logger.error(f"[{request_id}] prediction failed: {e}")
            log_prediction(
                request_id=request_id, model_version=MODEL_VERSION, latency_ms=latency_ms,
                status="error", raw_input=student.model_dump(), error=str(e),
            )
            raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")
        latency_ms = elapsed()

    prediction = round(prediction, 1)
    metrics.record_prediction(latency_ms)
    logger.info(f"[{request_id}] predicted_final_score={prediction} latency_ms={latency_ms:.1f}")
    log_prediction(
        request_id=request_id, model_version=MODEL_VERSION, latency_ms=latency_ms,
        status="success", raw_input=student.model_dump(), predicted_final_score=prediction,
    )

    return PredictionOutput(
        predicted_final_score=prediction, model_version=MODEL_VERSION, request_id=request_id,
    )
