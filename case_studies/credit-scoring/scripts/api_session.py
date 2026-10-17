"""A guided tour of the running API: health, valid / missing-income / invalid requests, metrics, and a latency benchmark.

Start the API first:   uvicorn src.api.app:app --port 8000
Then:                  python scripts/api_session.py --url http://localhost:8000 [--n 200]

Latency numbers: p50 = the median request, p95 = the slowest 5% start here.  Tiny example: 100 timings sorted, p95 = the 95th value.
These are LOCAL numbers (one process, one laptop, requests made one after another) - good for comparing before/after a change,
not a capacity plan.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import argparse
import json
import statistics
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=200, help="requests in the latency benchmark")
a = ap.parse_args()
samples = json.loads(Path("configs/sample_requests.json").read_text())


def show(title, r):
    print(f"\n--- {title}  ->  HTTP {r.status_code}")
    body = r.json()
    if isinstance(body, dict) and "reason_codes" in body:
        print(f"    probability_of_default={body['probability_of_default']}  risk_band={body['risk_band']}  decision_hint={body['decision_hint']}  model_version={body['model_version']}")
        for rc in body["reason_codes"]:
            print(f"      reason: {rc['reason']}  (contribution {rc['contribution']:+.2f})")
    else:
        print("    " + json.dumps(body)[:300])


print("GET /health");        print("   ", requests.get(f"{a.url}/health", timeout=5).json())
print("GET /model-info");    print("   ", json.dumps(requests.get(f"{a.url}/model-info", timeout=5).json())[:400])
for k in ("low_risk", "medium_risk", "high_risk", "missing_income"):
    show(f"POST /predict  [{k}]", requests.post(f"{a.url}/predict", json=samples[k], timeout=10))
show("POST /predict  [INVALID: loan_amount = -5]", requests.post(f"{a.url}/predict", json={**samples["low_risk"], "loan_amount": -5}, timeout=10))
show("POST /predict  [INVALID: unknown category employment_type='Salaried']", requests.post(f"{a.url}/predict", json={**samples["low_risk"], "employment_type": "Salaried"}, timeout=10))
show("POST /predict  [REJECTED: sensitive field gender is not part of the contract]", requests.post(f"{a.url}/predict", json={**samples["low_risk"], "gender": "female"}, timeout=10))
show("POST /predict  [REJECTED: leaky field interest_rate_offered]", requests.post(f"{a.url}/predict", json={**samples["low_risk"], "interest_rate_offered": 12.5}, timeout=10))

# ---- latency benchmark ----------------------------------------------------------------------------------------------
sess = requests.Session()
for _ in range(10):                                           # warm-up (first calls are slower)
    sess.post(f"{a.url}/predict", json=samples["medium_risk"], timeout=10)
times = []
for _ in range(a.n):
    t0 = time.perf_counter()
    sess.post(f"{a.url}/predict", json=samples["medium_risk"], timeout=10)
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
lat = dict(n=a.n, mean_ms=round(statistics.mean(times), 1), p50_ms=round(times[len(times) // 2], 1),
           p95_ms=round(times[int(len(times) * 0.95) - 1], 1), max_ms=round(times[-1], 1))
print(f"\nLATENCY over {a.n} sequential requests: {lat}")
Path("reports").mkdir(exist_ok=True)
Path("reports/api_latency.json").write_text(json.dumps(lat, indent=2))

print("\nGET /metrics (excerpt)")
for line in requests.get(f"{a.url}/metrics", timeout=5).text.splitlines():
    if line.startswith(("api_requests_total", "model_predictions_total", "model_info", "api_prediction_errors_total")) or \
            line.startswith('api_request_latency_seconds_count{endpoint="/predict"}'):
        print("   ", line)
