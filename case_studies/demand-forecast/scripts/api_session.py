"""Start the API on a free port, run a scripted session (what you would do with curl), print everything, stop the API.

    python scripts/api_session.py [--history data/processed/sales_history.csv] [--n 200]

Shows: /health, a valid prediction, invalid requests (HTTP 422), an unknown series, a stale-history request, a /metrics excerpt and a
small latency benchmark (p50 / p95 of /predict).  Used to capture docs/sample_outputs/09_api_session.txt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import argparse
import json
import os
import socket
import subprocess
import time

import numpy as np
import requests

from src.common.facts import save_fact

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--history", default=None)
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
env = dict(os.environ, **({"HISTORY_PATH": a.history} if a.history else {}))
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env)
url = f"http://127.0.0.1:{port}"
try:
    for _ in range(60):
        try:
            requests.get(f"{url}/health", timeout=1); break
        except requests.ConnectionError:
            time.sleep(0.5)
    sample = json.loads((ROOT / "configs" / "sample_request.json").read_text())
    show = lambda title, r: print(f"\n$ {title}\nHTTP {r.status_code}  {json.dumps(r.json(), indent=None)[:420]}")
    show("GET /health", requests.get(f"{url}/health"))
    show("POST /predict   (valid)  " + json.dumps(sample), requests.post(f"{url}/predict", json=sample))
    show("POST /predict   (promotion: price cut 61 -> 49)", requests.post(f"{url}/predict", json={**sample, "promotion": 1, "discount": 12.0, "price": 49.0}))
    show("POST /predict   (weather forecast missing)", requests.post(f"{url}/predict", json={k: v for k, v in sample.items() if k != "forecast_temp"}))
    show("POST /predict   (2 days stale: forecast_date 2025-03-03)", requests.post(f"{url}/predict", json={**sample, "forecast_date": "2025-03-03"}))
    show("POST /predict   (INVALID: price 61000 = unit error, > 5000)", requests.post(f"{url}/predict", json={**sample, "price": 61000}))
    show("POST /predict   (INVALID: missing fields)", requests.post(f"{url}/predict", json={"nonsense": 1}))
    show("POST /predict   (unknown product P99)", requests.post(f"{url}/predict", json={**sample, "product_id": "P99"}))
    show("POST /predict   (history too stale: 2025-04-01)", requests.post(f"{url}/predict", json={**sample, "forecast_date": "2025-04-01"}))
    show("GET /model-info", requests.get(f"{url}/model-info"))
    # latency benchmark
    lat = []
    for i in range(a.n):
        body = {**sample, "forecast_date": str(np.datetime64("2025-01-10") + int(i % 40)), "product_id": f"P0{1 + i % 8}", "store_id": f"S{1 + i % 3}"}
        t0 = time.perf_counter(); r = requests.post(f"{url}/predict", json=body); lat.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200, r.text
    lat = np.array(lat)
    print(f"\nLATENCY /predict over {a.n} sequential requests (client side, includes HTTP): p50 = {np.percentile(lat, 50):.1f} ms, "
          f"p95 = {np.percentile(lat, 95):.1f} ms, max = {lat.max():.1f} ms")
    text = requests.get(f"{url}/metrics").text
    print("\n$ GET /metrics   (excerpt)")
    for line in text.splitlines():
        if line.startswith(("api_requests_total", "model_predictions_total", "model_info", "api_prediction_errors_total")) or "model_score_count" in line:
            print(" ", line)
    save_fact("api_latency", dict(n=a.n, p50_ms=round(float(np.percentile(lat, 50)), 1), p95_ms=round(float(np.percentile(lat, 95)), 1), max_ms=round(float(lat.max()), 1)))
finally:
    proc.terminate(); proc.wait(timeout=10)
