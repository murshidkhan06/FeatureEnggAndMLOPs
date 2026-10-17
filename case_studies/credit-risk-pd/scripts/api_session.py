"""Start the API on a free port, run a scripted session (what you would do with curl), print everything, stop the API.

    python scripts/api_session.py [--n 200]

Shows: /health, a valid prediction (a customer with a clean record and one with warning signs), nullable fields, invalid requests (HTTP 422),
/model-info, a /metrics excerpt and a small latency benchmark (p50 / p95 of /predict).  Used to capture docs/sample_outputs/09_api_session.txt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import argparse
import json
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import requests

from src.common.facts import save_fact

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT)
url = f"http://127.0.0.1:{port}"
try:
    for _ in range(60):
        try:
            requests.get(f"{url}/health", timeout=1); break
        except requests.ConnectionError:
            time.sleep(0.5)
    sample = json.loads((ROOT / "configs" / "sample_request.json").read_text())
    risky = {**sample, "customer_id": 77, "features_as_of": "2024-01-31", "average_balance": 31000.0, "balance_trend": -9000.0, "salary_credit_frequency": 0.5,
             "loan_payment_ratio": 0.5, "missed_payment_count": 3, "credit_utilization": 0.95, "recent_transaction_count": 2, "income_stability": 0.31}
    no_card = {**sample, "customer_id": 78, "credit_utilization": None}

    def show(title, r):
        print(f"\n$ {title}\nHTTP {r.status_code}  {json.dumps(r.json())[:520]}")

    show("GET /health", requests.get(f"{url}/health"))
    show("POST /predict   (valid, typical record: card utilisation 51 %)  " + json.dumps(sample)[:90] + "...", requests.post(f"{url}/predict", json=sample))
    show("POST /predict   (valid, warning signs: 3 missed payments, utilisation 95 %, salary every other month)", requests.post(f"{url}/predict", json=risky))
    show("POST /predict   (customer without a credit card: credit_utilization = null -> imputed exactly as in training)", requests.post(f"{url}/predict", json=no_card))
    show("POST /predict   (INVALID: credit_utilization 7.5 > 3.0)", requests.post(f"{url}/predict", json={**sample, "credit_utilization": 7.5}))
    show("POST /predict   (INVALID: missed_payment_count 9 > 6 published months)", requests.post(f"{url}/predict", json={**sample, "missed_payment_count": 9}))
    show("POST /predict   (INVALID: no features_as_of -> the timestamp is part of the data contract)", requests.post(f"{url}/predict", json={k: v for k, v in sample.items() if k != "features_as_of"}))
    show("POST /predict   (INVALID: garbage)", requests.post(f"{url}/predict", json={"nonsense": 1}))
    show("GET /model-info", requests.get(f"{url}/model-info"))
    batch = pd.read_csv(ROOT / "data" / "production" / "batch_normal.csv").head(a.n)
    fields = ["customer_id", "features_as_of", "average_balance", "balance_trend", "salary_credit_frequency", "loan_payment_ratio",
              "missed_payment_count", "credit_utilization", "recent_transaction_count", "income_stability", "account_age"]
    lat = []
    for _, row in batch.iterrows():
        body = {k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k])) for k in fields}
        t0 = time.perf_counter(); r = requests.post(f"{url}/predict", json=body); lat.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200, r.text
    lat = np.array(lat)
    print(f"\nLATENCY /predict over {len(lat)} sequential requests (client side, includes HTTP): p50 = {np.percentile(lat, 50):.1f} ms, "
          f"p95 = {np.percentile(lat, 95):.1f} ms, max = {lat.max():.1f} ms")
    text = requests.get(f"{url}/metrics").text
    print("\n$ GET /metrics   (excerpt)")
    for line in text.splitlines():
        if line.startswith(("api_requests_total", "model_predictions_total", "model_info", "api_prediction_errors_total")) or "model_score_count" in line:
            print(" ", line)
    save_fact("api_latency", dict(n=len(lat), p50_ms=round(float(np.percentile(lat, 50)), 1), p95_ms=round(float(np.percentile(lat, 95)), 1), max_ms=round(float(lat.max()), 1)))
finally:
    proc.terminate(); proc.wait(timeout=10)
