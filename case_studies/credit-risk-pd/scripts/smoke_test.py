"""Call a RUNNING API (local, Docker or Minikube) and check it behaves.
python scripts/smoke_test.py --url http://localhost:8000 --sample configs/sample_request.json"""
import argparse
import json
import sys

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--sample", default="configs/sample_request.json")
a = ap.parse_args()
h = requests.get(f"{a.url}/health", timeout=5).json()
print("health:", h)
assert h["model_loaded"], "model not loaded"
r = requests.post(f"{a.url}/predict", json=json.load(open(a.sample)), timeout=10)
print("predict:", r.status_code, r.json())
assert r.status_code == 200
bad = requests.post(f"{a.url}/predict", json={"unexpected": 1}, timeout=10)
print("invalid request ->", bad.status_code)
assert bad.status_code == 422
print("SMOKE TEST PASSED")
sys.exit(0)
