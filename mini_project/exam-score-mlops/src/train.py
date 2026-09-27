"""
src/train.py

The single, reproducible entrypoint for training this project's model. Everything that
was, in the original notebook, spread across "run this cell, then this cell, then this
cell" now happens as one deterministic run driven entirely by params.yaml.

Run from the repo root:
    python -m src.train

What it does, in order (mirrors PHASE 8 of the course spec):
    1. Load params.yaml
    2. Load + validate data (src/data_processing.py)
    3. Split (before any preprocessing -- leakage-safe, per the notebook's Section 2)
    4. Build the feature+model pipeline (src/pipeline_builder.py, reused from the notebook)
    5. Train
    6. Evaluate on the held-out test set
    7. Log parameters, metrics, and artifacts to MLflow
    8. Save the fitted pipeline as models/exam_score_pipeline.joblib
    9. Write a human-readable evaluation report to reports/
    10. Register the model in the MLflow Model Registry

IMPORTANT: this script does not print training-serving-skew warnings, because there's
nothing to warn about -- api/main.py loads the EXACT artifact this script saves, and never
duplicates any of the feature engineering above.
"""
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import load_raw_data, split_features_target, train_test_split_data
from src.evaluate import evaluate_pipeline, feature_coefficients
from src.pipeline_builder import build_pipeline


def load_params(params_path: Path = REPO_ROOT / "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def train(params: dict | None = None, register: bool = True) -> dict:
    params = params or load_params()

    # ---- 1-3: data -------------------------------------------------------------------
    df = load_raw_data(REPO_ROOT / params["data"]["raw_path"])
    X, y = split_features_target(
        df, params["data"]["target_column"], params["data"]["id_column"]
    )
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, params["split"]["test_size"], params["split"]["random_state"]
    )

    # ---- 4-5: build + fit --------------------------------------------------------------
    pipeline = build_pipeline(params)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        # ---- 6: evaluate -----------------------------------------------------------
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        coef_df = feature_coefficients(pipeline)

        # ---- 7: log to MLflow -------------------------------------------------------
        mlflow.log_params({
            "model_type": params["model"]["type"],
            "lasso_alpha": params["feature_selection"]["lasso_alpha"],
            "test_size": params["split"]["test_size"],
            "random_state": params["split"]["random_state"],
            "pca_n_components": params["pca"]["n_components"],
            "numeric_impute_strategy": params["preprocessing"]["numeric_impute_strategy"],
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
        })
        mlflow.log_metrics(metrics)

        REPO_ROOT.joinpath("reports").mkdir(exist_ok=True)
        coef_path = REPO_ROOT / "reports" / "feature_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)
        mlflow.log_artifact(str(coef_path))

        report_path = REPO_ROOT / "reports" / "evaluation_report.json"
        report = {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "n_features_kept": len(coef_df),
            "top_features": coef_df.head(5).to_dict(orient="records"),
        }
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        # log the fitted pipeline as an MLflow model (enables the registry below).
        # NOTE: MLflow's newer default serializer (skops) refuses to save a pipeline that
        # contains a custom class (our FeatureCreator) unless you explicitly mark it
        # trusted -- a genuinely useful safety default. We use the classic pickle-based
        # format instead (same underlying idea as the joblib file we save below), which is
        # exactly the PHASE 7 "serialization risk: trusted artifacts only" lesson in
        # practice, not just in the README.
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=params["mlflow"]["registered_model_name"] if register else None,
            serialization_format="pickle",
        )

        # ---- 8: save the joblib artifact the API actually loads ---------------------
        REPO_ROOT.joinpath("models").mkdir(exist_ok=True)
        model_path = REPO_ROOT / "models" / "exam_score_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        # ---- reference distribution for drift detection ------------------------------
        # Saves the RAW (pre-feature-engineering) training inputs as the "what training
        # data looked like" snapshot. src/drift.py compares production traffic against
        # THIS file, never against the live training script -- exactly what a real
        # reference distribution is: a frozen snapshot, not a moving target.
        REPO_ROOT.joinpath("data", "reference").mkdir(parents=True, exist_ok=True)
        reference_path = REPO_ROOT / "data" / "reference" / "training_reference.csv"
        X_train.to_csv(reference_path, index=False)

        print(f"MLflow run: {run.info.run_id}")
        print(f"R2={metrics['r2']:.3f}  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")
        print(f"Saved pipeline -> {model_path}")
        print(f"Saved evaluation report -> {report_path}")

    return {"run_id": run.info.run_id, "metrics": metrics, "model_path": str(model_path)}


if __name__ == "__main__":
    train()
