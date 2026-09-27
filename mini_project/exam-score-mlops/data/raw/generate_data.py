"""
data/raw/generate_data.py

Generates `student_exam_scores.csv` -- the single dataset used throughout this project.
Reused, unchanged in logic, from the original teaching notebook's data generator
(mini_project/exam_score_project/data/raw/generate_data.py) so the production pipeline is
trained on the SAME data the notebook was built and explained against.

Deliberately built to justify every feature engineering concept the notebook demonstrates:
- missing values in study_hours (imputation)
- three correlated mock test scores (PCA)
- a genuine interaction effect between study_hours and attendance_pct (feature creation)
- a mix of ordinal and nominal categorical columns (encoding)
- an enrollment_date column (time-based feature creation)
- a couple of weak/noisy columns (feature selection has something real to drop)

Run from the repo root:
    python data/raw/generate_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 500
SNAPSHOT_DATE = pd.Timestamp("2026-01-01")

study_hours = np.clip(rng.normal(10, 4, N), 0, 30)
attendance_pct = np.clip(rng.normal(75, 15, N), 30, 100)

# Three mock tests, all driven by the same underlying "ability" factor -- genuinely correlated
ability = rng.normal(60, 15, N)
mock_test_1 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
mock_test_2 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
mock_test_3 = np.clip(ability + rng.normal(0, 6, N), 0, 100)

income_bracket = rng.choice(["Low", "Medium", "High"], size=N, p=[0.35, 0.45, 0.20])
city = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Pune"], size=N)

enrollment_date = SNAPSHOT_DATE - pd.to_timedelta(rng.integers(30, 730, N), unit="D")

# Weak / noisy columns -- feature selection should learn to de-prioritize these
shoe_size = rng.normal(8, 1.5, N)          # genuinely irrelevant to exam performance
lucky_number = rng.integers(1, 100, N)      # pure noise

income_effect = {"Low": -2.0, "Medium": 0.0, "High": 3.0}

# TRUE relationship: study_hours and attendance_pct have a genuine INTERACTION --
# studying helps much more when attendance is also high (makes real-world sense: showing up
# to class is what makes the studying "click"). This is exactly why the interaction feature
# we engineer later actually earns its place, not just decoration.
interaction_true_effect = 0.015 * study_hours * attendance_pct

final_score = (
    25
    + 1.1 * study_hours
    + 0.25 * attendance_pct
    + interaction_true_effect
    + 0.28 * ability
    + np.array([income_effect[b] for b in income_bracket])
    + rng.normal(0, 4, N)
)
final_score = np.clip(final_score, 0, 100).round(1)

df = pd.DataFrame({
    "student_id": np.arange(1, N + 1),
    "study_hours": study_hours.round(1),
    "attendance_pct": attendance_pct.round(1),
    "mock_test_1": mock_test_1.round(1),
    "mock_test_2": mock_test_2.round(1),
    "mock_test_3": mock_test_3.round(1),
    "income_bracket": income_bracket,
    "city": city,
    "enrollment_date": enrollment_date,
    "shoe_size": shoe_size.round(1),
    "lucky_number": lucky_number,
    "final_score": final_score,
})

# Realistic missingness in study_hours only (MAR-ish: busier/less-engaged students skip logging it)
missing_idx = rng.choice(N, size=int(N * 0.06), replace=False)
df.loc[missing_idx, "study_hours"] = np.nan

df = df.sample(frac=1, random_state=7).reset_index(drop=True)

OUT_PATH = Path(__file__).resolve().parent / "student_exam_scores.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH} with shape {df.shape}")
print(f"Missing study_hours: {df['study_hours'].isna().sum()} rows")
print(df[["mock_test_1", "mock_test_2", "mock_test_3"]].corr().round(2))
