"""The model artifact = the whole sklearn Pipeline (preprocessing + model) + metadata.

Saving the PIPELINE (not just the estimator) is what prevents training-serving skew:
the exact same fitted imputers/scalers/encoders run at prediction time.
"""
from __future__ import annotations

from datetime import datetime, timezone

import joblib


def save_artifact(path, pipeline, version: str, input_columns: list[str], metrics: dict,
                  threshold: float = 0.5, extra: dict | None = None) -> None:
    joblib.dump(dict(pipeline=pipeline, version=version, input_columns=input_columns, metrics=metrics,
                     threshold=threshold, extra=extra or {},
                     trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds")), path)


def load_artifact(path) -> dict:
    return joblib.load(path)
