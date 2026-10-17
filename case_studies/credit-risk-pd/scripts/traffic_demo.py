"""The monitoring loop on your laptop:  API -> prediction log -> drift exporter -> KS numbers.

    python scripts/traffic_demo.py [--n 400]

1. start the API, send NORMAL production traffic (data/production/batch_normal.csv, calm month 2023-03)
2. run ONE update of the drift exporter (what Prometheus would scrape) and print the KS gauges
3. repeat with DRIFTED traffic (batch_drifted.csv, stress month 2024-01)
Every request is logged (logs/predictions.jsonl) with the MODEL INPUT features (never the customer_id); the exporter compares the last N logged
rows with data/processed/reference.csv.  A flag means 'the INPUTS look different' - the decision to retrain still needs outcome evidence
(python -m src.monitoring.run_drift_check).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import argparse
import socket
import subprocess
import time

import requests

from src import project as P
from src.common.facts import save_fact
from src.common.paths import PREDICTION_LOG
from src.monitoring import drift_exporter as X

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()
facts: dict = {}


def run(label: str, batch: str) -> None:
    PREDICTION_LOG.unlink(missing_ok=True)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                requests.get(f"{url}/health", timeout=1); break
            except requests.ConnectionError:
                time.sleep(0.5)
        print(f"\n=== {label}: {batch} ===")
        subprocess.run([sys.executable, "scripts/send_traffic.py", "--csv", batch, "--n", str(a.n), "--shuffle", "--url", url], cwd=ROOT, check=True)
        text = requests.get(f"{url}/metrics").text
        print("predictions by risk band:", [l.split("{")[1] for l in text.splitlines() if l.startswith("model_predictions_total")])
    finally:
        proc.terminate(); proc.wait(timeout=10)
    X.update(window=a.n, alpha=0.05, min_effect=0.10)
    print(f"drift exporter gauges (window = last {int(X.ROWS._value.get())} logged requests):")
    print(f"  {'feature':<26}{'KS D':>8}{'p-value':>11}  flag")
    rows = {}
    for f in P.DRIFT_FEATURES:
        d, p, fl = X.KS.labels(f)._value.get(), X.PVAL.labels(f)._value.get(), int(X.FLAG.labels(f)._value.get())
        rows[f] = dict(ks=round(d, 4), p=float(p), flag=fl)
        print(f"  {f:<26}{d:8.3f}{p:11.2g}  {fl}")
    facts[label] = rows


run("NORMAL traffic", "data/production/batch_normal.csv")
run("DRIFTED traffic", "data/production/batch_drifted.csv")
save_fact("traffic_demo", facts)
PREDICTION_LOG.unlink(missing_ok=True)
