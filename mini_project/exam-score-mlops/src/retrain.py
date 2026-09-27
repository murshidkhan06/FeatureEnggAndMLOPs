"""
src/retrain.py

Retraining + promotion (PHASE 20): retrain a CANDIDATE model on (possibly updated) data,
evaluate it on identical terms to the current CHAMPION (production) model, and promote the
candidate ONLY if it is actually better -- never automatically, and never just because
drift was detected.

    NEW DATA -> VALIDATE -> RETRAIN (candidate) -> EVALUATE
             -> COMPARE with CHAMPION -> PROMOTE only if better -> DEPLOY (swap joblib)

WHY this gate matters: "drift detected" tells you the INPUT distribution changed. It does
NOT tell you the new model is better -- a retrain on a small, noisy batch of new data can
easily be WORSE than the existing champion. This script's promotion rule is deliberately
conservative:

    promote candidate IFF
        candidate_mae <= champion_mae - params.retraining.min_mae_improvement
        AND candidate_mae <= params.evaluation.max_acceptable_mae

Run from the repo root:
    python -m src.retrain
    python -m src.retrain --data-path data/raw/student_exam_scores.csv   # or a new CSV
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "exam_score_pipeline.joblib"

import sys
sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import load_raw_data, split_features_target, train_test_split_data
from src.evaluate import evaluate_pipeline
from src.pipeline_builder import build_pipeline


def load_params() -> dict:
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def evaluate_champion(champion_path: Path, X_test, y_test) -> dict | None:
    """Evaluates the CURRENTLY DEPLOYED model on the SAME held-out split as the
    candidate, so the comparison is apples-to-apples. Returns None if there is no
    champion yet (first-ever training run)."""
    if not champion_path.exists():
        return None
    champion = joblib.load(champion_path)
    return evaluate_pipeline(champion, X_test, y_test)


def retrain(data_path: Path | None = None, params: dict | None = None) -> dict:
    params = params or load_params()
    data_path = data_path or (REPO_ROOT / params["data"]["raw_path"])

    # ---- validate + split (same discipline as src/train.py) --------------------------
    df = load_raw_data(data_path)
    X, y = split_features_target(df, params["data"]["target_column"], params["data"]["id_column"])
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, params["split"]["test_size"], params["split"]["random_state"]
    )

    # ---- evaluate the CURRENT champion on this split, before touching anything -------
    champion_metrics = evaluate_champion(MODEL_PATH, X_test, y_test)

    # ---- train the candidate -----------------------------------------------------------
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="retrain_candidate") as run:
        candidate = build_pipeline(params)
        candidate.fit(X_train, y_train)
        candidate_metrics = evaluate_pipeline(candidate, X_test, y_test)

        mlflow.log_params({
            "run_type": "retrain_candidate",
            "data_path": str(data_path),
            "lasso_alpha": params["feature_selection"]["lasso_alpha"],
        })
        mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_metrics.items()})
        if champion_metrics:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})

        model_info = mlflow.sklearn.log_model(
            candidate, artifact_path="model", serialization_format="pickle",
            registered_model_name=params["mlflow"]["registered_model_name"],
        )

        # ---- promotion gate ------------------------------------------------------------
        max_mae = params["evaluation"]["max_acceptable_mae"]
        min_improvement = params["retraining"]["min_mae_improvement"]

        passes_quality_bar = candidate_metrics["mae"] <= max_mae
        if champion_metrics is None:
            is_better = True  # nothing to compare against yet -- first model ever
            reason = "no existing champion to compare against"
        else:
            improvement = champion_metrics["mae"] - candidate_metrics["mae"]
            is_better = improvement >= min_improvement
            reason = (
                f"MAE improved by {improvement:.3f} "
                f"(champion={champion_metrics['mae']:.3f}, candidate={candidate_metrics['mae']:.3f}, "
                f"required>={min_improvement})"
            )

        promoted = passes_quality_bar and is_better
        mlflow.log_param("promoted", promoted)
        mlflow.set_tag("promotion_reason", reason)

        client = MlflowClient()
        registered_version = model_info.registered_model_version

        if promoted:
            joblib.dump(candidate, MODEL_PATH)  # this is what api/main.py loads next restart
            client.set_registered_model_alias(
                params["mlflow"]["registered_model_name"], "champion", registered_version
            )
            decision = "PROMOTED"
        else:
            client.set_registered_model_alias(
                params["mlflow"]["registered_model_name"], "rejected_candidate", registered_version
            )
            decision = "REJECTED"

        print(f"\n[{decision}] candidate MLflow version {registered_version}")
        print(f"  candidate metrics: {candidate_metrics}")
        print(f"  champion metrics:  {champion_metrics}")
        print(f"  quality bar (MAE <= {max_mae}): {'PASS' if passes_quality_bar else 'FAIL'}")
        print(f"  reason: {reason}")

        result = {
            "run_id": run.info.run_id,
            "decision": decision,
            "candidate_metrics": candidate_metrics,
            "champion_metrics": champion_metrics,
        }
        report_path = REPO_ROOT / "reports" / "retrain_report.json"
        report_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(report_path))

    return result


def main():
    parser = argparse.ArgumentParser(description="Retrain a candidate model and promote it if it's better.")
    parser.add_argument("--data-path", type=Path, default=None,
                         help="CSV to retrain on (default: params.yaml's data.raw_path)")
    args = parser.parse_args()
    retrain(data_path=args.data_path)


if __name__ == "__main__":
    main()
