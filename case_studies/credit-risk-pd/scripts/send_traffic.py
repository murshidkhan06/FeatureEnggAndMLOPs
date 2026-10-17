"""Send requests from a production CSV to a running API (to generate metrics / logs / drift).

python scripts/send_traffic.py --csv data/production/batch_normal.csv --n 300 [--url http://localhost:8000]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import argparse
import time

import pandas as pd
import requests

from src import project as P

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--sleep", type=float, default=0.0)
ap.add_argument("--shuffle", action="store_true", help="send a random sample instead of the first n rows")
a = ap.parse_args()

fields = list(P.PredictRequest.model_fields)
df = pd.read_csv(a.csv)
df = df.sample(min(a.n, len(df)), random_state=0) if a.shuffle else df.head(a.n)
ok = bad = 0
t0 = time.time()
for _, row in df.iterrows():
    payload = {k: (None if pd.isna(row[k]) else row[k].item() if hasattr(row[k], "item") else row[k]) for k in fields}
    payload = {k: v for k, v in payload.items() if v is not None}
    r = requests.post(f"{a.url}/predict", json=payload, timeout=10)
    ok += r.status_code == 200
    bad += r.status_code != 200
    if a.sleep:
        time.sleep(a.sleep)
print(f"sent={ok + bad} ok={ok} errors={bad} in {time.time() - t0:.1f}s")
