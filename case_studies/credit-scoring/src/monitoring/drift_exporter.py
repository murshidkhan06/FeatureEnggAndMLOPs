"""Tiny Prometheus exporter: compares the last N logged predictions with the reference data (KS test)
and exposes the result as gauges that Grafana can plot.

Run:  python -m src.monitoring.drift_exporter --port 9100 --window 500 --interval 30
      python -m src.monitoring.drift_exporter --once            (one update, print the table, exit: handy for testing)
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


def update(window: int, alpha: float, min_effect: float):
    """Compare the last `window` logged predictions with the reference; set the gauges; return the KS report (or None)."""
    if not PREDICTION_LOG.exists() or not REFERENCE_PATH.exists():
        return None
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
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-effect", type=float, default=0.10)
    ap.add_argument("--once", action="store_true", help="update once, print the table and exit (no HTTP server)")
    a = ap.parse_args()
    if a.once:
        rep = update(a.window, a.alpha, a.min_effect)
        print("no prediction log yet (send some traffic first)" if rep is None else
              f"drift exporter: last {a.window} logged predictions vs reference\n" + rep.to_string(index=False))
        raise SystemExit(0)
    start_http_server(a.port)
    print(f"drift exporter on :{a.port}")
    while True:
        update(a.window, a.alpha, a.min_effect)
        time.sleep(a.interval)
