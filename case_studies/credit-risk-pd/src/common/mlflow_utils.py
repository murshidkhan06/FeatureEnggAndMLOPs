"""Thin helpers so every project logs to MLflow the same way (local sqlite store, no server needed).

View the UI with:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import contextlib
import os

import mlflow

from .paths import ROOT


def setup(experiment: str) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


@contextlib.contextmanager
def tracked_run(run_name: str, params: dict, tags: dict | None = None):
    """with tracked_run('baseline', params) as run:  mlflow.log_metric(...)"""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)
        yield run
