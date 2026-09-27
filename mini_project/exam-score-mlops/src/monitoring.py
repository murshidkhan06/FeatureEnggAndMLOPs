"""
src/monitoring.py

Observability for the deployed model: structured logging, a prediction log, and simple
in-process metrics -- all local files, no external monitoring stack required (PHASE 15 /
PHASE 16 of the course spec explicitly keep this optional-Prometheus, local-first).

WHY log predictions at all? Two reasons that matter in production:
    1. If something goes wrong (a bad prediction, a user complaint), you need to be able
       to look up exactly what input produced it -- "prediction logs" are your audit trail.
    2. Drift detection (src/drift.py) needs a "current production distribution" to compare
       against the training distribution -- that distribution comes from these logs.

WHAT gets logged, and what deliberately does NOT: every raw feature value used for
prediction, plus the prediction, latency, and a model version tag. We do NOT log anything
that isn't already part of the model's own input schema -- no incidental PII, matching
PHASE 15's "do not log sensitive information unnecessarily."

THE PREDICTION STORE (extended here for Unit 4): `predictions.csv` alone only tells you
WHAT the model predicted -- it says nothing about whether that prediction was any good,
because the true `final_score` for a real student isn't known at prediction time (that's
the whole point of predicting it!). It typically becomes known LATER: after the actual
exam happens, once grades are entered into the college's system, days or weeks after the
prediction was made. This module adds a second small store, `labels.csv`, for exactly that
DELAYED LABEL: `record_actual_label()` appends a (request_id, actual_final_score) row
whenever a true outcome becomes available, and `src/performance_monitor.py` joins the two
files on `request_id` to compute real accuracy (MAE/RMSE/R2) -- this is what makes MODEL
performance-decay monitoring possible, as distinct from DATA drift monitoring (src/drift.py),
which needs no labels at all and can run the instant a request comes in.
"""
import csv
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MONITORING_DIR = REPO_ROOT / "monitoring"
PREDICTIONS_LOG = MONITORING_DIR / "predictions.csv"
LABELS_LOG = MONITORING_DIR / "labels.csv"
APP_LOG = MONITORING_DIR / "app.log"

PREDICTION_LOG_COLUMNS = [
    "timestamp", "request_id", "model_version", "latency_ms", "status",
    "study_hours", "attendance_pct", "mock_test_1", "mock_test_2", "mock_test_3",
    "income_bracket", "city", "enrollment_date", "shoe_size", "lucky_number",
    "predicted_final_score", "error",
]

# The PREDICTION STORE, in this teaching project, is simply these two flat CSV files
# (predictions.csv + labels.csv), joined on request_id -- see the module docstring below
# and src/performance_monitor.py for why they're kept as two separate files rather than
# one, and how "delayed labels" are modeled.
LABEL_LOG_COLUMNS = ["timestamp", "request_id", "actual_final_score"]


def _configure_logger() -> logging.Logger:
    MONITORING_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("exam_score_api")
    if logger.handlers:  # avoid duplicate handlers on reload
        return logger
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(APP_LOG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(stream_handler)
    return logger


logger = _configure_logger()


class Metrics:
    """Simple in-process counters. Intentionally NOT persisted across restarts -- for
    anything that needs to survive a restart or be queried historically, predictions.csv
    (on disk) is the source of truth, not these in-memory counters."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self._latencies_ms: list[float] = []

    def record_request(self) -> None:
        with self._lock:
            self.request_count += 1

    def record_prediction(self, latency_ms: float) -> None:
        with self._lock:
            self.prediction_count += 1
            self._latencies_ms.append(latency_ms)

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = (
                sum(self._latencies_ms) / len(self._latencies_ms)
                if self._latencies_ms else 0.0
            )
            return {
                "request_count": self.request_count,
                "prediction_count": self.prediction_count,
                "error_count": self.error_count,
                "error_rate": (
                    self.error_count / self.request_count if self.request_count else 0.0
                ),
                "avg_latency_ms": round(avg_latency, 2),
            }


metrics = Metrics()


@contextmanager
def timed():
    """Usage: with timed() as t: ... ; t() -> elapsed milliseconds."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


def log_prediction(
    request_id: str,
    model_version: str,
    latency_ms: float,
    status: str,
    raw_input: Optional[Dict[str, Any]] = None,
    predicted_final_score: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Appends one row to monitoring/predictions.csv. Creates the file with a header on
    first use. This is the file src/drift.py and src/monitoring's own report read back."""
    MONITORING_DIR.mkdir(exist_ok=True)
    file_exists = PREDICTIONS_LOG.exists()

    raw_input = raw_input or {}
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": request_id,
        "model_version": model_version,
        "latency_ms": round(latency_ms, 2),
        "status": status,
        "predicted_final_score": predicted_final_score,
        "error": error,
        **{k: raw_input.get(k) for k in PREDICTION_LOG_COLUMNS if k in raw_input},
    }

    with open(PREDICTIONS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def record_actual_label(
    request_id: str,
    actual_final_score: float,
    timestamp: Optional[str] = None,
) -> None:
    """Appends one row to monitoring/labels.csv -- the DELAYED LABEL for a prediction made
    earlier. In this teaching project this is called by `POST /feedback` (api/main.py); in
    a real deployment it would typically be called by a batch job that reconciles actual
    outcomes (e.g. real exam results loaded from the college's system) against past
    `request_id`s on some regular cadence, not synchronously per-request."""
    MONITORING_DIR.mkdir(exist_ok=True)
    file_exists = LABELS_LOG.exists()
    row = {
        "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": request_id,
        "actual_final_score": actual_final_score,
    }
    with open(LABELS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_predictions_log(path: Path = PREDICTIONS_LOG) -> Dict[str, Any]:
    """A quick CLI-friendly summary of the prediction log -- used by `python -m
    src.monitoring` and referenced by the README's monitoring walkthrough step."""
    if not path.exists():
        return {"total_predictions": 0, "message": "No predictions logged yet."}

    import pandas as pd
    df = pd.read_csv(path)
    return {
        "total_predictions": len(df),
        "success_count": int((df["status"] == "success").sum()),
        "error_count": int((df["status"] == "error").sum()),
        "avg_latency_ms": round(df["latency_ms"].mean(), 2) if len(df) else 0.0,
        "avg_predicted_score": (
            round(df["predicted_final_score"].dropna().mean(), 2)
            if df["predicted_final_score"].notna().any() else None
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summarize_predictions_log(), indent=2))
