"""
kubeflow/components.py

KUBEFLOW PIPELINE COMPONENTS for the Exam Score MLOps project.

WHY Kubeflow at all, when src/pipeline.py (Unit 3) already runs training end to end?
`python -m src.pipeline` is a single Python process running a fixed sequence of Python
function calls -- great for training alone, but it cannot express things a REAL production
ML system needs: "run this step in its own container/environment", "only run the next step
IF this one's output says to" (conditional retraining), "run this on a schedule, on a
cluster, with retries, with each step's inputs/outputs tracked as first-class artifacts you
can inspect after the fact". That's what an ORCHESTRATOR is for. Kubeflow Pipelines (KFP)
is the orchestrator this project uses: it runs an ML workflow as a DAG of independent
COMPONENTS, each with declared INPUTS, OUTPUTS, and PARAMETERS, designed to run on
Kubernetes in production -- see kubeflow/run_local.py and kubeflow/compile_pipeline.py for
how the SAME pipeline definition below runs locally (for this course) and would run on a
real cluster (in production).

HOW each component below is implemented: every component is a THIN WRAPPER that shells out
to this project's EXISTING, ALREADY-TESTED CLI entrypoints (`python -m src.train`,
`python -m src.drift`, etc.) via `subprocess.run(..., cwd=repo_root)`, then reads back the
JSON/CSV report each of those scripts already writes to `reports/`. WHY wrap instead of
reimplement: every one of those scripts is already a leakage-safe, tested, working piece of
this project (Unit 1-3 and the earlier Unit 4 modules) -- reimplementing their logic INSIDE
a Kubeflow component would mean maintaining the same logic twice, and risking the two
copies quietly drifting apart. Wrapping is also exactly how a real team typically
"Kubeflow-izes" an existing codebase: the ML logic doesn't move, only the ORCHESTRATION
around it changes.

DESIGN DECISION worth calling out explicitly: the course's reference pipeline diagram lists
Data Preparation, Feature Engineering, Training, Evaluation, and MLflow Logging as separate
boxes. Here they are deliberately implemented as ONE component (`train_and_evaluate`), not
five. WHY: those five steps must run ATOMICALLY inside a single, leakage-safe
`sklearn.Pipeline.fit()` call (see src/pipeline_builder.py) -- splitting "feature
engineering" and "training" into separately-schedulable Kubeflow components would mean the
scaler/PCA/feature-selector could get fit on a different slice of data than the model, which
is exactly the kind of train/test leakage this whole project has been careful to avoid since
Unit 1. Not every box in a reference architecture diagram should become its own DAG node --
sometimes the right call is "these steps are one atomic unit of work, and that unit is what
becomes the component." Evaluation and MLflow logging are included in the SAME component for
the same reason: `src/train.py` evaluates and logs to MLflow inside the SAME `mlflow.start_
run()` context that fit the model, so a run's params/metrics/artifacts are guaranteed to
correspond to exactly one fit -- splitting evaluation into a separate component would need
either re-fitting the model (wasteful) or passing the fitted pipeline object between
components (fragile pickling across process/container boundaries) for no real benefit.

ARTIFACTS AND PARAMETERS in play: `repo_root` (a PARAMETER, str) tells every component
where the project lives; each component's typed return values (`r2: float`, `decision:
str`, `data_drift_detected: bool`, ...) are Kubeflow PARAMETER outputs, automatically
passed to downstream components and to the pipeline's conditional logic (see
kubeflow/pipeline.py's `dsl.If`); the actual heavyweight files each step produces (the
trained `.joblib` model, `reports/*.json`/`*.csv`) are left on disk under `repo_root`
(exactly where every other part of this project already expects to find them) rather than
passed as KFP ARTIFACTS between components -- a deliberate simplification for the LOCAL
teaching setup (see kubeflow/run_local.py's docstring for why, and how a production
Kubeflow deployment would instead use real `Output[Artifact]`/`Input[Artifact]` objects
backed by cloud object storage).
"""
import json
import subprocess
import sys
from typing import NamedTuple

from kfp import dsl


def _run_module(repo_root: str, module: str, extra_args: list[str] | None = None) -> str:
    """Shared helper: runs `python -m <module> [extra_args]` with cwd=repo_root, using the
    SAME Python interpreter this component is executing under (so it sees this project's
    installed dependencies, whether that's a venv locally or the component's container
    environment on a real cluster). Raises with the full stderr on failure -- a Kubeflow
    component that fails should fail LOUDLY, with enough detail in the pipeline's logs to
    debug it, never silently.
    """
    result = subprocess.run(
        [sys.executable, "-m", module, *(extra_args or [])],
        cwd=repo_root, capture_output=True, text=True,
    )
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"`python -m {module}` failed with exit code {result.returncode}")
    return result.stdout


@dsl.component(base_image="python:3.11")
def data_validation_component(repo_root: str) -> str:
    """COMPONENT 1: Data Validation.
    INPUTS:  repo_root (parameter) -- where the project lives.
    OUTPUTS: status (parameter, str) -- "OK", or the component fails loudly (see
             src/data_processing.py's validate_raw_data -- required columns present,
             target column in range with no missing values, categorical values as
             expected). This mirrors the "fail fast, before anything downstream wastes
             time on bad data" principle already used throughout src/data_processing.py.
    """
    import subprocess, sys as _sys
    code = (
        "from src.data_processing import load_raw_data; "
        "import yaml; "
        "params = yaml.safe_load(open('params.yaml')); "
        "df = load_raw_data(params['data']['raw_path']); "
        "print(f'Validated {len(df)} rows -- OK')"
    )
    result = subprocess.run([_sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("Data validation failed -- see logs above.")
    return "OK"


@dsl.component(base_image="python:3.11")
def train_and_evaluate_component(repo_root: str) -> NamedTuple(
    "Outputs", model_path=str, mlflow_run_id=str, r2=float, mae=float, rmse=float
):
    """COMPONENT 2: Data Preparation + Feature Engineering + Training + Evaluation +
    MLflow Logging (deliberately ONE component -- see this module's docstring for why).
    INPUTS:  repo_root (parameter).
    OUTPUTS: model_path (parameter, str) -- where the fitted pipeline was saved.
             mlflow_run_id (parameter, str) -- the MLflow run this training logged to;
                 this is the integration point with MLflow (see this module's docstring
                 and the README's "MLflow + Kubeflow" section: Kubeflow ORCHESTRATES this
                 step, MLflow TRACKS what happened inside it -- Kubeflow does not replace
                 MLflow's experiment tracking or model registry).
             r2, mae, rmse (parameter, float) -- held-out test-set metrics, read back from
                 reports/evaluation_report.json (written by src/train.py).
    """
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path

    result = _subprocess.run([_sys.executable, "-m", "src.train"], cwd=repo_root, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("Training failed -- see logs above.")

    report = _json.loads((_Path(repo_root) / "reports" / "evaluation_report.json").read_text())
    metrics = report["metrics"]
    model_path = str(_Path(repo_root) / "models" / "exam_score_pipeline.joblib")

    from typing import NamedTuple as _NamedTuple
    Outputs = _NamedTuple("Outputs", model_path=str, mlflow_run_id=str, r2=float, mae=float, rmse=float)
    return Outputs(
        model_path=model_path, mlflow_run_id=report["run_id"],
        r2=metrics["r2"], mae=metrics["mae"], rmse=metrics["rmse"],
    )


@dsl.component(base_image="python:3.11")
def drift_detection_component(repo_root: str) -> NamedTuple(
    "Outputs", data_drift_detected=bool, n_numeric_drifted=int, n_categorical_drifted=int
):
    """COMPONENT: Drift Detection (numeric KS test + categorical chi-square test, see
    src/drift.py). Requires monitoring/predictions.csv to already have some production
    traffic in it (see src/generate_traffic.py) -- in a real deployment this component
    would run on a schedule against whatever traffic has accumulated since it last ran.
    INPUTS:  repo_root (parameter).
    OUTPUTS: data_drift_detected (parameter, bool) -- used by the pipeline's conditional
                 logic (see kubeflow/pipeline.py) to decide whether to even bother checking
                 for a retraining trigger.
             n_numeric_drifted, n_categorical_drifted (parameter, int).
    """
    import csv as _csv
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path

    predictions_log = _Path(repo_root) / "monitoring" / "predictions.csv"
    if not predictions_log.exists():
        # No production traffic yet (e.g. right after a fresh deployment) is a normal,
        # expected state -- NOT a pipeline failure. Nothing to compare against yet, so
        # there is, correctly, nothing to flag.
        print("No production traffic in monitoring/predictions.csv yet -- skipping drift "
              "check for this run (nothing to compare against).")
        from typing import NamedTuple as _NamedTuple
        Outputs = _NamedTuple("Outputs", data_drift_detected=bool, n_numeric_drifted=int, n_categorical_drifted=int)
        return Outputs(data_drift_detected=False, n_numeric_drifted=0, n_categorical_drifted=0)

    result = _subprocess.run([_sys.executable, "-m", "src.drift"], cwd=repo_root, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("Drift detection failed -- see logs above.")

    def _count_drifted(path):
        p = _Path(repo_root) / "reports" / path
        if not p.exists():
            return 0
        with open(p) as f:
            return sum(1 for row in _csv.DictReader(f) if row.get("drift") == "True")

    n_numeric = _count_drifted("drift_report.csv")
    n_categorical = _count_drifted("categorical_drift_report.csv")

    from typing import NamedTuple as _NamedTuple
    Outputs = _NamedTuple("Outputs", data_drift_detected=bool, n_numeric_drifted=int, n_categorical_drifted=int)
    return Outputs(
        data_drift_detected=bool(n_numeric or n_categorical),
        n_numeric_drifted=n_numeric, n_categorical_drifted=n_categorical,
    )


@dsl.component(base_image="python:3.11")
def retrain_trigger_decision_component(repo_root: str) -> str:
    """COMPONENT: Retraining Trigger Decision (see src/retrain_decision.py's full
    docstring for the decision table this wraps). This is the component whose output the
    PIPELINE branches on -- see kubeflow/pipeline.py's `dsl.If(decision == "RETRAIN")`.
    Performance decay (needs delayed labels) is the deciding factor; data drift alone is
    treated as evidence to watch, not evidence to act on -- see src/retrain_decision.py.

    IMPLEMENTATION NOTE: returns a single plain `str` (the decision) rather than a
    NamedTuple with a separate `reason` field. `kfp`'s LOCAL runner (as of kfp 2.17) has a
    known limitation resolving a `dsl.If` condition that references one named field of a
    multi-output component (it silently fails to enter the branch at all, with no error --
    only a `Could not resolve parent input` warning). A single-output component's `.output`
    does not have this problem, on the local runner OR a real cluster. The full `reason`
    string is still printed to this component's logs (and written to
    reports/retrain_decision_report.json) -- nothing is lost, only how it's WIRED to the
    next component changes.

    INPUTS:  repo_root (parameter).
    OUTPUTS: decision (str) -- "RETRAIN" or "NO_RETRAIN".
    """
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path

    result = _subprocess.run([_sys.executable, "-m", "src.retrain_decision"], cwd=repo_root, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("Retrain-decision check failed -- see logs above.")

    report = _json.loads((_Path(repo_root) / "reports" / "retrain_decision_report.json").read_text())
    print(f"reason: {report['reason']}")
    return report["decision"]


@dsl.component(base_image="python:3.11")
def retrain_and_promote_component(repo_root: str, data_path: str = "") -> str:
    """COMPONENT: Retrain candidate + Model Promotion Decision (see src/retrain.py's
    promotion gate: candidate must beat the champion's MAE by a configured margin AND
    clear an absolute quality bar -- never promoted just because drift or a trigger fired).
    Only reached when the pipeline's `dsl.If(decision == "RETRAIN")` branch is taken.

    `data_path` (optional): retrain on a SPECIFIC (presumably updated/new) CSV instead of
    params.yaml's default raw_path. WHY this matters: retraining on the exact same data the
    current champion was already trained on can never legitimately clear the promotion
    gate's min-improvement bar (there's nothing new to learn -- see Scenario 4 in
    scripts/classroom_demo.py). A retraining trigger that fired because of confirmed
    performance decay (e.g. real-world concept drift) only has a chance of producing a
    genuinely BETTER, promotable candidate (Scenario 5) when it is retrained on data that
    reflects the NEW reality -- exactly what a real "new data -> retrain" pipeline run
    would be pointed at.

    IMPLEMENTATION NOTE: single `str` output for the same reason as
    `retrain_trigger_decision_component` above -- see that component's docstring. The
    candidate's full metrics are still printed to this component's logs and written to
    reports/retrain_report.json.

    INPUTS:  repo_root (parameter), data_path (parameter, optional).
    OUTPUTS: promotion_decision (str) -- "PROMOTED" or "REJECTED".
    """
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path

    args = [_sys.executable, "-m", "src.retrain"]
    if data_path:
        args += ["--data-path", data_path]
    result = _subprocess.run(args, cwd=repo_root, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("Retraining failed -- see logs above.")

    report = _json.loads((_Path(repo_root) / "reports" / "retrain_report.json").read_text())
    print(f"candidate_mae: {report['candidate_metrics']['mae']}")
    return report["decision"]


@dsl.component(base_image="python:3.11")
def deployment_gate_component(repo_root: str, promotion_decision: str, run_docker_build: bool = False) -> str:
    """COMPONENT 9 (optional): Deployment gate. Only reached when a candidate was actually
    PROMOTED. Deliberately does NOT restart the live FastAPI process or push to Docker Hub
    automatically -- consistent with this project's existing "training pipeline and
    application deployment pipeline are kept decoupled" design (see src/pipeline.py's
    docstring and README section 3/26): a retrain-and-promote should never silently and
    automatically take down a running production container. With `run_docker_build=True`
    and a local Docker daemon available, it WILL build a fresh image (a bounded, clearly
    logged action) -- but pushing/redeploying that image remains a separate, manual step.
    INPUTS:  repo_root (parameter), promotion_decision (parameter, str, from the previous
                 component), run_docker_build (parameter, bool, default False).
    OUTPUTS: status (parameter, str).
    """
    import subprocess as _subprocess
    import shutil as _shutil

    if promotion_decision != "PROMOTED":
        print(f"promotion_decision={promotion_decision!r} -- nothing to deploy.")
        return "SKIPPED"

    print("A new champion model was promoted (models/exam_score_pipeline.joblib updated).")
    print("Next, MANUAL steps (deliberately not automated -- see this component's docstring):")
    print("  1. docker build -t <you>/exam-score-api:<new-tag> .")
    print("  2. docker push <you>/exam-score-api:<new-tag>")
    print("  3. Roll out the new image to wherever the API is actually running.")

    if run_docker_build:
        if _shutil.which("docker") is None:
            print("run_docker_build=True but no local Docker daemon found -- skipping the build.")
            return "PROMOTED_NO_DOCKER"
        result = _subprocess.run(
            ["docker", "build", "-t", "exam-score-api:kubeflow-candidate", "."],
            cwd=repo_root, capture_output=True, text=True,
        )
        print(result.stdout[-2000:])
        if result.returncode != 0:
            print(result.stderr[-2000:])
            return "PROMOTED_DOCKER_BUILD_FAILED"
        print("Docker image built: exam-score-api:kubeflow-candidate (push/deploy left manual).")
        return "PROMOTED_DOCKER_BUILT"

    return "PROMOTED_MANUAL_DEPLOY_PENDING"
