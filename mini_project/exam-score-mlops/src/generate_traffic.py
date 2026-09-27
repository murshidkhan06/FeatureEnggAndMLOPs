"""
src/generate_traffic.py

Generates realistic (or deliberately drifted) prediction traffic against a RUNNING
FastAPI instance -- this is what fills monitoring/predictions.csv with enough rows for
src/drift.py's KS test and src/retrain.py's promotion logic to be meaningful, instead of
the handful of rows a couple of manual curl calls would leave behind.

Prerequisite: the API must already be running in another terminal:
    uvicorn api.main:app --port 8000

Run from the repo root:
    python -m src.generate_traffic --n 200
    python -m src.generate_traffic --n 200 --simulate-drift   # shifts study/attendance down
"""
import argparse
import random
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "student_exam_scores.csv"


def sample_students(n: int, simulate_drift: bool, seed: int = 123) -> pd.DataFrame:
    """Samples `n` synthetic requests. With --simulate-drift, shifts study_hours and
    attendance_pct downward -- a realistic scenario (e.g. a new, less-engaged cohort, or
    exam season fatigue) that src/drift.py should actually be able to catch."""
    rng = random.Random(seed)
    base_df = pd.read_csv(DATA_PATH, parse_dates=["enrollment_date"])
    rows = []
    for _ in range(n):
        row = base_df.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]
        payload = {
            "study_hours": float(row["study_hours"]) if pd.notna(row["study_hours"]) else 8.0,
            "attendance_pct": float(row["attendance_pct"]),
            "mock_test_1": float(row["mock_test_1"]),
            "mock_test_2": float(row["mock_test_2"]),
            "mock_test_3": float(row["mock_test_3"]),
            "income_bracket": row["income_bracket"],
            "city": row["city"],
            "enrollment_date": str(row["enrollment_date"].date()),
            "shoe_size": float(row["shoe_size"]),
            "lucky_number": int(row["lucky_number"]),
        }
        if simulate_drift:
            payload["study_hours"] = max(0.0, payload["study_hours"] * 0.5)
            payload["attendance_pct"] = max(0.0, payload["attendance_pct"] * 0.7)
        rows.append(payload)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Send simulated prediction traffic to a running API.")
    parser.add_argument("--n", type=int, default=200, help="number of requests to send")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--simulate-drift", action="store_true",
                         help="shift study_hours/attendance_pct downward to create real, detectable drift")
    args = parser.parse_args()

    payloads = sample_students(args.n, args.simulate_drift)

    sent, failed = 0, 0
    with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
        for _, payload in payloads.iterrows():
            try:
                response = client.post("/predict", json=payload.to_dict())
                response.raise_for_status()
                sent += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"Request failed: {e}")

    print(f"Sent {sent} requests successfully, {failed} failed "
          f"({'drift-simulated' if args.simulate_drift else 'normal'} traffic).")
    print("Check monitoring/predictions.csv, then run: python -m src.drift")


if __name__ == "__main__":
    main()
