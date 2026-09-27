"""
src/pipeline.py

The ONE command that runs the complete ML TRAINING pipeline end to end (PHASE 22):

    DATA VALIDATION -> FEATURE ENGINEERING (inside train) -> TRAIN -> EVALUATE
    -> MLFLOW LOGGING -> MODEL ARTIFACT -> TEST

Deliberately STOPS there. Docker build/deploy is a separate, later step (see the
Makefile's `deploy` target and the README's Docker section) -- this project keeps the
TRAINING pipeline and the APPLICATION DEPLOYMENT pipeline decoupled on purpose (PHASE 22),
so retraining a model never accidentally also rebuilds and redeploys a container, and vice
versa.

Run from the repo root:
    python -m src.pipeline
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(description: str, func) -> None:
    print(f"\n{'=' * 70}\nSTEP: {description}\n{'=' * 70}")
    func()


def step_validate_data():
    from src.data_processing import load_raw_data
    import yaml
    with open(REPO_ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)
    df = load_raw_data(REPO_ROOT / params["data"]["raw_path"])
    print(f"Validated {len(df)} rows -- OK")


def step_train_and_evaluate():
    from src.train import train
    result = train()
    print(f"Trained + logged to MLflow (run_id={result['run_id']})")
    print(f"Metrics: {result['metrics']}")


def step_run_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(
            "Tests failed after training -- treat this as a pipeline failure, not a "
            "green build. Fix the failing test(s) before trusting this model."
        )
    print("All tests passed.")


def main():
    run_step("1/3 Data validation", step_validate_data)
    run_step("2/3 Train + evaluate + log to MLflow + save artifact", step_train_and_evaluate)
    run_step("3/3 Run test suite against the freshly trained model", step_run_tests)
    print("\nPipeline complete. Model is trained, evaluated, logged, and tested.")
    print("Next (separate, manual steps): `uvicorn api.main:app` to serve it locally, or "
          "`docker build` to package it -- see the README.")


if __name__ == "__main__":
    main()
