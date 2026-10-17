"""The monitoring loop on your laptop:  API -> prediction log -> drift exporter -> KS numbers.

    python scripts/traffic_demo.py [--n 400]

1. start the API (normal history), send NORMAL production traffic (random days of data/production/batch_normal.csv)
2. run ONE update of the drift exporter (what Prometheus would scrape) and print the KS gauges
3. repeat with the DRIFTED history + batch_drifted.csv
Every request is logged (logs/predictions.jsonl) with the model-input features; the exporter compares the last N logged rows with
data/processed/reference.csv.  Reading the result: a flag means 'the INPUTS look different' - the decision to retrain still needs the
labelled-performance evidence (python -m src.monitoring.run_drift_check).  Seasonal remark: January-February weather differs from the
all-year reference, so forecast_temp is flagged even for NORMAL traffic - an expected seasonal shift, not a failure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import argparse
import os
import socket
import subprocess
import time

import requests

from src import project as P
from src.common.paths import PREDICTION_LOG
from src.monitoring import drift_exporter as X

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()


def run(label: str, batch: str, history: str) -> None:
    PREDICTION_LOG.unlink(missing_ok=True)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=dict(os.environ, HISTORY_PATH=history))
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                requests.get(f"{url}/health", timeout=1); break
            except requests.ConnectionError:
                time.sleep(0.5)
        print(f"\n=== {label}: {batch}  (history: {history}) ===")
        subprocess.run([sys.executable, "scripts/send_traffic.py", "--csv", batch, "--n", str(a.n), "--shuffle", "--url", url], cwd=ROOT, check=True)
        text = requests.get(f"{url}/metrics").text
        print("predictions by demand band:", [l.split("{")[1] for l in text.splitlines() if l.startswith("model_predictions_total")])
    finally:
        proc.terminate(); proc.wait(timeout=10)
    X.update(window=a.n, alpha=0.05, min_effect=0.10)
    print(f"drift exporter gauges (window = last {int(X.ROWS._value.get())} logged requests):")
    print(f"  {'feature':<16}{'KS D':>8}{'p-value':>11}  flag")
    for f in P.DRIFT_FEATURES:
        print(f"  {f:<16}{X.KS.labels(f)._value.get():8.3f}{X.PVAL.labels(f)._value.get():11.2g}  {int(X.FLAG.labels(f)._value.get())}")


run("NORMAL traffic", "data/production/batch_normal.csv", "data/processed/sales_history.csv")
run("DRIFTED traffic", "data/production/batch_drifted.csv", "data/processed/sales_history_drifted.csv")
PREDICTION_LOG.unlink(missing_ok=True)
