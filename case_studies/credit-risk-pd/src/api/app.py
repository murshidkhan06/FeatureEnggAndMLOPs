"""FastAPI serving layer (shared by every project).  Project specifics live in src/project.py.

Endpoints
  POST /predict     score one record (request/response are pydantic models -> automatic validation)
  GET  /health      liveness + whether a model is loaded
  GET  /model-info  which model version is serving
  GET  /metrics     Prometheus text format (request count, latency, errors, prediction mix)
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src import project as P
from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, PREDICTION_LOG

REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["risk_level"])
SCORE = Histogram("model_score", "Predicted probability", buckets=[i / 10 for i in range(11)])
MODEL_INFO = Gauge("model_info", "Serving model version (value is always 1)", ["version"])
ERRORS = Counter("api_prediction_errors_total", "Unhandled errors inside /predict")

STATE: dict = {"artifact": None}


def load_model(path: str | None = None) -> None:
    path = Path(path or os.getenv("MODEL_PATH", CHAMPION_PATH))
    if path.exists():
        STATE["artifact"] = load_artifact(path)
        MODEL_INFO.labels(version=STATE["artifact"]["version"]).set(1)
    else:
        STATE["artifact"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title=P.TITLE, version="1.0.0", lifespan=lifespan,
              description="Educational project. See README for the data and the limits of this model.")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ep = request.url.path
        if ep not in ("/predict", "/health", "/metrics", "/model-info"):
            ep = "other"
        REQUESTS.labels(ep, request.method, str(status)).inc()
        LATENCY.labels(ep).observe(time.perf_counter() - start)


@app.get("/health")
def health():
    art = STATE["artifact"]
    return {"status": "ok" if art else "degraded", "model_loaded": art is not None,
            "model_version": art["version"] if art else None}


@app.get("/model-info")
def model_info():
    art = STATE["artifact"]
    if not art:
        raise HTTPException(503, "No model loaded")
    return {"model_version": art["version"], "trained_at": art["trained_at"],
            "metrics": art["metrics"], "threshold": art["threshold"], "input_columns": art["input_columns"]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _log_prediction(record: dict, result: dict) -> None:
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                **record, **result}, default=str) + "\n")
    except OSError:
        pass  # logging must never break serving


@app.post("/predict", response_model=P.PredictResponse)
def predict(req: P.PredictRequest):
    art = STATE["artifact"]
    if not art:
        raise HTTPException(503, "Model not loaded")
    try:
        frame = P.request_to_frame(req)              # raw request -> model input row(s)
        proba = float(art["pipeline"].predict_proba(frame[art["input_columns"]])[:, 1][0])
    except Exception as exc:                         # noqa: BLE001
        ERRORS.inc()
        raise HTTPException(500, f"Prediction failed: {exc}") from exc
    level = P.risk_level(proba)
    PREDICTIONS.labels(level).inc()
    SCORE.observe(proba)
    result = P.build_response(req, proba, level, art)
    _log_prediction(frame.iloc[0].to_dict(), {"probability": round(proba, 6), "risk_level": level,
                                              "model_version": art["version"]})
    return result
