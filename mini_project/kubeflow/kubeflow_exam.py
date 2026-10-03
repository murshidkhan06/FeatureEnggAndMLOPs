"""Exam-score training pipeline for Kubeflow Pipelines (kfp v2).

Steps: prepare_data -> train_model -> evaluate_model -> (if good enough) approve_model

Run:
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80   # in another terminal
    python kubeflow_pipeline.py                                    # compiles and submits a run
    python kubeflow_pipeline.py --compile-only                     # writes the YAML only
"""
import argparse

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

BASE_IMAGE = "python:3.11-slim"
PACKAGES = ["pandas==2.2.3", "scikit-learn==1.5.2", "joblib==1.4.2"]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def prepare_data(n_students: int, train_data: Output[Dataset], test_data: Output[Dataset]):
    """Create a synthetic student dataset and split it 80/20."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "study_hours": rng.uniform(0, 10, n_students).round(1),
        "attendance_pct": rng.uniform(50, 100, n_students).round(1),
        "mock_test_avg": rng.uniform(30, 95, n_students).round(1),
    })
    noise = rng.normal(0, 4, n_students)
    df["final_score"] = (2.5 * df.study_hours + 0.3 * df.attendance_pct
                         + 0.45 * df.mock_test_avg + noise).clip(0, 100).round(1)

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv(train_data.path, index=False)
    test.to_csv(test_data.path, index=False)
    print(f"train={len(train)} rows, test={len(test)} rows")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def train_model(train_data: Input[Dataset], alpha: float, model: Output[Model]):
    """Fit a scaler + Lasso pipeline and save it as model.joblib."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(train_data.path)
    X, y = df.drop(columns="final_score"), df["final_score"]
    pipe = Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=alpha))])
    pipe.fit(X, y)
    joblib.dump(pipe, model.path)
    model.metadata["alpha"] = alpha
    model.metadata["framework"] = "scikit-learn"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def evaluate_model(test_data: Input[Dataset], model: Input[Model], metrics: Output[Metrics]) -> float:
    """Score the model on the held-out test set; return MAE."""
    import joblib
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    df = pd.read_csv(test_data.path)
    X, y = df.drop(columns="final_score"), df["final_score"]
    pipe = joblib.load(model.path)
    pred = pipe.predict(X)
    mae = float(mean_absolute_error(y, pred))
    metrics.log_metric("mae", round(mae, 3))
    metrics.log_metric("r2", round(float(r2_score(y, pred)), 3))
    return mae


@dsl.component(base_image=BASE_IMAGE)
def approve_model(mae: float, threshold: float):
    """Quality gate passed: in a real system, register the model as champion here."""
    print(f"APPROVED: MAE {mae:.2f} is below threshold {threshold}")


@dsl.pipeline(name="exam-score-training", description="Prepare data, train, evaluate and gate an exam-score model")
def exam_score_pipeline(n_students: int = 500, alpha: float = 0.1, mae_threshold: float = 5.0):
    data = prepare_data(n_students=n_students)
    trained = train_model(train_data=data.outputs["train_data"], alpha=alpha)
    evaluated = evaluate_model(test_data=data.outputs["test_data"], model=trained.outputs["model"])
    with dsl.If(evaluated.outputs["Output"] < mae_threshold, name="mae-below-threshold"):
        approve_model(mae=evaluated.outputs["Output"], threshold=mae_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    compiler.Compiler().compile(exam_score_pipeline, "exam_score_pipeline.yaml")
    print("Compiled -> exam_score_pipeline.yaml")

    if not args.compile_only:
        import kfp

        client = kfp.Client(host=args.host)
        run = client.create_run_from_pipeline_func(
            exam_score_pipeline,
            arguments={"n_students": 500, "alpha": 0.1, "mae_threshold": 5.0},
            experiment_name="exam-score",
        )
        print(f"Run submitted: {args.host}/#/runs/details/{run.run_id}")