"""
src/generate_batch.py

Generates a standalone CSV batch of "new" student records -- WITHOUT needing a running
FastAPI instance. This exists specifically so an Airflow DAG task can call it directly
(as a plain Python function, or `python -m src.generate_batch` as a subprocess) to produce
a `current` dataset for `src/drift.py` to compare against the training-time reference,
without wiring up an API + traffic generator inside the DAG just to get a CSV.

WHY a separate module from src/generate_traffic.py? That module deliberately simulates
REAL production traffic against a live API (useful for the FastAPI/monitoring story). This
module skips the API entirely and writes a CSV straight to disk -- the right shape for an
orchestrated batch pipeline step, where "the next task reads a file" is the natural
hand-off, not "make 150 HTTP calls".

Run from the repo root:
    python -m src.generate_batch --n 150 --out data/scenarios/new_batch.csv
    python -m src.generate_batch --n 150 --drift --out data/scenarios/new_batch.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "student_exam_scores.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "scenarios" / "new_batch.csv"


def generate_batch(n: int = 150, drift: bool = False, seed: int = 123) -> pd.DataFrame:
    """Samples `n` rows from the existing raw dataset (with replacement -- this is meant
    to represent NEW incoming records, not a held-out split of the original data).

    With drift=True, shifts study_hours and attendance_pct downward -- the same realistic
    scenario src/generate_traffic.py uses (e.g. a less-engaged new cohort, or exam-season
    fatigue): a real, teachable data drift the KS test in src/drift.py should catch.
    """
    rng = random.Random(seed)
    base_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["enrollment_date"])
    idx = [rng.randrange(len(base_df)) for _ in range(n)]
    batch = base_df.iloc[idx].reset_index(drop=True).copy()

    if drift:
        batch["study_hours"] = (batch["study_hours"] * 0.55).round(1)
        batch["attendance_pct"] = (batch["attendance_pct"] * 0.8).clip(upper=100).round(1)

    return batch


def main():
    parser = argparse.ArgumentParser(description="Generate a standalone batch CSV for drift checks.")
    parser.add_argument("--n", type=int, default=150, help="Number of rows to generate.")
    parser.add_argument("--drift", action="store_true", help="Simulate a shifted (drifted) cohort.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    batch = generate_batch(n=args.n, drift=args.drift, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(args.out, index=False)
    print(f"Wrote {len(batch)} rows -> {args.out}  (drift={'ON' if args.drift else 'off'})")


if __name__ == "__main__":
    main()
