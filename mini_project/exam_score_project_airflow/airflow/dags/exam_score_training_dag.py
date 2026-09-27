"""
airflow/dags/exam_score_training_dag.py

UNIT 4 (ORCHESTRATION, Airflow variant): the SAME exam_score_project pipeline this course
already orchestrates with a plain Python script (`src/pipeline.py`, Units 1-3) and with
Kubeflow Pipelines (`kubeflow/pipeline.py`, in the companion Monitoring & Kubeflow module)
-- expressed here as an Apache Airflow DAG.

WHY Airflow, when Kubeflow already exists? They solve overlapping but different problems:

  - Airflow's core strength is TIME-BASED, GENERAL-PURPOSE workflow scheduling: "run this
    every Sunday at 2am, retry twice on failure, alert someone if it doesn't finish by
    9am" -- and it's equally at home orchestrating a data-warehouse ETL job as an ML
    pipeline. It is NOT ML-native: it has no built-in concept of "model", "metric", or
    "experiment run".
  - Kubeflow Pipelines is ML-native and Kubernetes-native: it understands artifacts and
    metrics as first-class pipeline outputs, and is built to run each step as its own
    container on a cluster.

This DAG deliberately reuses the SAME tested CLI modules the rest of this course already
relies on (src/train.py, src/drift.py, src/retrain.py) rather than reimplementing any of
their logic -- exactly the same "wrap, don't rewrite" principle the Kubeflow components use
(see kubeflow/components.py in the companion module). Every Airflow task below is a thin
subprocess wrapper around a command you could just as easily run by hand from a terminal.

THE DAG:

    validate_data -> train_and_evaluate -> generate_new_batch -> check_drift
        -> decide_retrain --[drift]--> retrain_model --> pipeline_summary
                          --[no drift]--> skip_retrain --> pipeline_summary

`decide_retrain` is a BranchPythonOperator (via the `@task.branch` TaskFlow decorator) --
Airflow's native mechanism for conditional branching, the direct analogue of Kubeflow's
`dsl.If`. Exactly as in the Kubeflow module: DRIFT DETECTED DOES NOT AUTOMATICALLY MEAN
RETRAIN in a full production design (a real decision should also weigh confirmed
performance decay and data volume -- see src/retrain_decision.py in the companion Unit 4
module for that fuller version). This DAG's branch is deliberately the SIMPLER,
drift-only version, to keep the Airflow-specific teaching point -- branching -- front and
center without re-deriving that whole decision table here.

Run locally (no cluster, no Docker -- Airflow's own local "standalone" mode):

    export AIRFLOW_HOME=$(pwd)/airflow
    pip install apache-airflow=="$(python -c 'import sys; print("2.9.3")')" --constraint ...
    airflow standalone            # starts scheduler + webserver + creates an admin user
    # UI at http://localhost:8080 -- trigger "exam_score_training_pipeline" manually

Or, to run and verify the DAG WITHOUT the webserver/scheduler at all (fastest for a
classroom demo or CI):

    airflow dags test exam_score_training_pipeline 2026-01-01
"""
from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

REPO_ROOT = Path(__file__).resolve().parents[2]

# IMPORTANT / COMMON MISTAKE this project deliberately avoids: Airflow's own Python
# environment does NOT automatically have this project's ML dependencies (pandas,
# scikit-learn, mlflow, ...) installed -- and pinning them into the SAME environment as
# Airflow is a common source of dependency conflicts in real deployments (Airflow ships
# its own strict `constraints-*.txt` pins for its own dependencies; MLflow's newer
# pydantic/typing-extensions/cryptography pins can conflict with them directly). The fix
# used throughout this project: Airflow gets its OWN isolated environment
# (.venv-airflow/), and every task shells out to the SEPARATE project environment
# (.venv/) that actually has the ML stack installed -- exactly what a real deployment
# does with a dedicated "task" virtualenv, a PythonVirtualenvOperator, or a per-task
# Docker/K8s image. See README.md Sec. "Two Environments, On Purpose".
PROJECT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if not PROJECT_PYTHON.exists():  # Windows, or a differently-named venv -- fall back
    PROJECT_PYTHON = Path(sys.executable)

default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _run_module(module: str, args: list[str] | None = None) -> str:
    """Runs `python -m <module> [args]` from the repo root, using the PROJECT's Python
    environment (not Airflow's) -- and raises loudly on failure, the same
    subprocess-wrapping pattern kubeflow/components.py uses, so a task's failure reason is
    never swallowed."""
    cmd = [str(PROJECT_PYTHON), "-m", module, *(args or [])]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"`{' '.join(cmd)}` failed with exit code {result.returncode}")
    return result.stdout


@dag(
    dag_id="exam_score_training_pipeline",
    description="Validate -> Train -> Evaluate -> Check Drift -> (conditional) Retrain",
    default_args=default_args,
    schedule="@weekly",  # a real cadence: retrain weekly, not "whenever someone remembers"
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["exam-score-project", "mlops", "orchestration"],
)
def exam_score_training_pipeline():

    @task
    def validate_data() -> None:
        """Fails loudly, before anything else runs, if the raw dataset doesn't look like
        what the pipeline expects (see src/data_processing.py's validate_raw_data)."""
        code = (
            "from src.data_processing import load_raw_data; import yaml; "
            "params = yaml.safe_load(open('params.yaml')); "
            "df = load_raw_data(params['data']['raw_path']); "
            "print(f'Validated {len(df)} rows -- OK')"
        )
        result = subprocess.run([str(PROJECT_PYTHON), "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError("Data validation failed")

    @task
    def train_and_evaluate() -> None:
        """python -m src.train -- feature engineering + model fit + evaluation + MLflow
        logging + reports/evaluation_report.json, all in one already-tested step."""
        _run_module("src.train")

    @task
    def generate_new_batch() -> None:
        """Stands in for 'a new batch of records arrived' -- deliberately drifted, so this
        DAG's branch has something real to demonstrate. See src/generate_batch.py."""
        _run_module("src.generate_batch", ["--n", "150", "--drift", "--out", "data/scenarios/new_batch.csv"])

    @task
    def check_drift() -> None:
        """python -m src.drift -- KS test, new batch vs. the training-time reference
        distribution. Writes reports/drift_report.csv for decide_retrain to read."""
        _run_module("src.drift", ["--current", "data/scenarios/new_batch.csv", "--numeric-only"])

    @task.branch
    def decide_retrain() -> str:
        """Airflow's native conditional-branching mechanism -- the direct analogue of
        Kubeflow's `dsl.If`. Returns the task_id of the ONE downstream task to run;
        Airflow automatically skips the other branch (visible as 'skipped', not 'failed',
        in the UI).

        Reads the CSV with the stdlib `csv` module rather than pandas on purpose -- this
        function runs inside AIRFLOW's own environment (the scheduler parses and executes
        DAG code directly), which deliberately does NOT have this project's ML stack
        installed. See the PROJECT_PYTHON note above."""
        report_path = REPO_ROOT / "reports" / "drift_report.csv"
        with open(report_path, newline="") as f:
            rows = list(csv.DictReader(f))
        drifted = [r["feature"] for r in rows if r["drift"] == "True"]
        if drifted:
            print(f"Drift detected in: {drifted} -> branching to retrain_model")
            return "retrain_model"
        print("No drift detected -> branching to skip_retrain")
        return "skip_retrain"

    @task
    def retrain_model() -> None:
        """python -m src.retrain --data-path ... -- trains a CANDIDATE on the new batch
        and promotes it over the current CHAMPION only if it clears the same promotion
        gate every candidate has to clear (see src/retrain.py)."""
        _run_module("src.retrain", ["--data-path", "data/scenarios/new_batch.csv"])

    @task
    def skip_retrain() -> None:
        print(
            "No retraining triggered this run. Note: a full production decision should "
            "also require CONFIRMED PERFORMANCE DECAY, not just drift -- see "
            "src/retrain_decision.py in the companion Unit 4 (Monitoring & Kubeflow) "
            "module for that fuller version. This DAG keeps the branch drift-only on "
            "purpose, to isolate the Airflow-specific lesson: native conditional "
            "branching with @task.branch."
        )

    @task(trigger_rule="none_failed_min_one_success")
    def pipeline_summary() -> None:
        print("exam_score_training_pipeline run complete.")

    v = validate_data()
    te = train_and_evaluate()
    batch = generate_new_batch()
    dc = check_drift()
    decision = decide_retrain()
    retrain = retrain_model()
    skip = skip_retrain()
    summary = pipeline_summary()

    v >> te >> batch >> dc >> decision
    decision >> [retrain, skip] >> summary


exam_score_training_pipeline()
