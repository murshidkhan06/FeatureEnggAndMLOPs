"""
src/batch_predict.py

Batch inference: CSV in -> predictions.csv out. Demonstrates PHASE 12's batch-vs-real-time
distinction using the SAME saved pipeline api/main.py uses for real-time inference --
there is no separate "batch preprocessing" logic, for the same training-serving-skew
reasons explained throughout this project.

WHEN to prefer batch over real-time:
    - You need predictions for many records at once (e.g. "score every currently-enrolled
      student tonight") and nobody is waiting on an individual HTTP response.
    - Throughput matters more than per-request latency -- batch amortizes model-loading
      and I/O cost across many rows instead of paying it once per request.
    - Real-time (FastAPI) matters when a single prediction is needed immediately in
      response to a user action, and low latency for THAT ONE request matters more than
      overall throughput.

Run from the repo root:
    python -m src.batch_predict --input data/raw/student_exam_scores.csv --output reports/batch_predictions.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src.predict import load_pipeline, predict_dataframe

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_batch_prediction(input_path: Path, output_path: Path, model_path: Path) -> pd.DataFrame:
    pipeline = load_pipeline(model_path)
    df = pd.read_csv(input_path)

    # The raw file may include columns the model was never trained on (e.g. student_id,
    # the true final_score for a labeled evaluation file) -- drop only what the pipeline
    # doesn't expect, keep everything else untouched for the output file.
    feature_cols = [
        "study_hours", "attendance_pct", "mock_test_1", "mock_test_2", "mock_test_3",
        "income_bracket", "city", "enrollment_date", "shoe_size", "lucky_number",
    ]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    predictions = predict_dataframe(pipeline, df[feature_cols])
    result = df.copy()
    result["predicted_final_score"] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch-predict exam scores for a CSV of students.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/raw/student_exam_scores.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports/batch_predictions.csv")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models/exam_score_pipeline.joblib")
    args = parser.parse_args()

    result = run_batch_prediction(args.input, args.output, args.model)
    print(f"Scored {len(result)} rows -> {args.output}")
    print(result[["predicted_final_score"]].describe())


if __name__ == "__main__":
    main()
