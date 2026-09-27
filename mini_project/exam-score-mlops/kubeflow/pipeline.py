"""
kubeflow/pipeline.py

The Kubeflow PIPELINE definition: wires the components in kubeflow/components.py into the
DAG the course spec asks for --

    DATA VALIDATION -> TRAIN+EVALUATE+MLFLOW -> DRIFT DETECTION -> RETRAIN TRIGGER DECISION
        -> [only if RETRAIN] RETRAIN CANDIDATE + PROMOTE
            -> [only if PROMOTED] DEPLOYMENT GATE

This is the SAME pipeline object used by both:
    kubeflow/run_local.py       -- executes it locally, no Kubernetes required (this course)
    kubeflow/compile_pipeline.py -- compiles it to pipeline.yaml, uploadable to a REAL
                                     Kubeflow cluster (production) -- see that script's
                                     docstring for exactly what would differ.

WHY the conditional branches (`dsl.If`), not just "always run everything": this is the
pipeline-level enforcement of this project's central Unit 4 teaching point -- DRIFT
DETECTED IS NOT THE SAME AS RETRAIN IMMEDIATELY. Retraining and promotion only execute at
all when `retrain_trigger_decision_component` says to; deployment only executes when the
resulting candidate was actually promoted. A pipeline run where nothing needed to change
completes successfully having done NOTHING beyond validate/train/check -- that is the
CORRECT, expected outcome for Scenario 1 (see notebooks/unit4_..._kubeflow.ipynb and
scripts/classroom_demo.py's five scenarios), not a partial failure.
"""
from kfp import dsl

from kubeflow.components import (
    data_validation_component,
    deployment_gate_component,
    drift_detection_component,
    retrain_and_promote_component,
    retrain_trigger_decision_component,
    train_and_evaluate_component,
)


@dsl.pipeline(
    name="exam-score-mlops-pipeline",
    description=(
        "Exam Score MLOps: validate -> train+evaluate+MLflow -> drift detection -> "
        "retrain-trigger decision -> [conditional] retrain+promote -> [conditional] deploy."
    ),
)
def exam_score_mlops_pipeline(repo_root: str, run_docker_build: bool = False, retrain_data_path: str = ""):
    validate_task = data_validation_component(repo_root=repo_root)
    validate_task.set_display_name("1. Data Validation")

    train_task = train_and_evaluate_component(repo_root=repo_root)
    train_task.after(validate_task)
    train_task.set_display_name("2. Train + Evaluate + MLflow Logging")

    drift_task = drift_detection_component(repo_root=repo_root)
    drift_task.after(train_task)
    drift_task.set_display_name("3. Drift Detection")

    decision_task = retrain_trigger_decision_component(repo_root=repo_root)
    decision_task.after(drift_task)
    decision_task.set_display_name("4. Retrain Trigger Decision")

    # "Drift detected != retrain immediately" -- retrain_trigger_decision_component already
    # weighed drift + performance decay + data volume (src/retrain_decision.py); the
    # pipeline itself only branches on its final RETRAIN / NO_RETRAIN verdict.
    #
    # NOTE: branches on `decision_task.output` (the component's single str return value),
    # not `decision_task.outputs["decision"]` -- see retrain_trigger_decision_component's
    # docstring in kubeflow/components.py for why: kfp's LOCAL runner has a known bug
    # resolving a dsl.If condition against one named field of a multi-output component.
    with dsl.If(decision_task.output == "RETRAIN", name="if-retrain-triggered"):
        retrain_task = retrain_and_promote_component(repo_root=repo_root, data_path=retrain_data_path)
        retrain_task.set_display_name("5. Retrain Candidate + Promotion Decision")

        # Deployment only proceeds if the candidate actually cleared the promotion gate --
        # a retrain that produces a WORSE model (Scenario 4) stops here, by design.
        with dsl.If(retrain_task.output == "PROMOTED", name="if-promoted"):
            deploy_task = deployment_gate_component(
                repo_root=repo_root,
                promotion_decision=retrain_task.output,
                run_docker_build=run_docker_build,
            )
            deploy_task.set_display_name("6. Deployment Gate")
