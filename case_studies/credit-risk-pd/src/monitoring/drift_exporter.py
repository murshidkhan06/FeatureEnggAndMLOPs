"""Tiny Prometheus exporter: compares the last N logged predictions with the reference data (KS test)
and exposes the result as gauges that Grafana can plot.

Run:  python -m src.monitoring.drift_exporter --port 9100 --window 500 --interval 30
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from prometheus_client import Gauge, start_http_server

from src import project as P
from src.common.drift import ks_drift_report
from src.common.paths import PREDICTION_LOG, REFERENCE_PATH

KS = Gauge("drift_ks_statistic", "KS statistic D (effect size)", ["feature"])
PVAL = Gauge("drift_p_value", "KS p-value", ["feature"])
FLAG = Gauge("drift_flag", "1 if significant AND practically large (status=drift)", ["feature"])
ROWS = Gauge("drift_production_rows", "Production rows compared")


def update(window: int, alpha: float, min_effect: float) -> None:
    if not PREDICTION_LOG.exists() or not REFERENCE_PATH.exists():
        return
    prod = pd.read_json(PREDICTION_LOG, lines=True).tail(window)
    ref = pd.read_csv(REFERENCE_PATH)
    feats = [f for f in P.DRIFT_FEATURES if f in prod.columns and f in ref.columns]
    rep = ks_drift_report(ref, prod, feats, alpha=alpha, min_effect=min_effect)
    ROWS.set(len(prod))
    for _, r in rep.iterrows():
        if r.status == "insufficient_data":
            continue
        KS.labels(r.feature).set(r.ks_statistic)
        PVAL.labels(r.feature).set(r.p_value)
        FLAG.labels(r.feature).set(1 if r.status == "drift" else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-effect", type=float, default=0.10)
    a = ap.parse_args()
    start_http_server(a.port)
    print(f"drift exporter on :{a.port}")
    while True:
        update(a.window, a.alpha, a.min_effect)
        time.sleep(a.interval)
