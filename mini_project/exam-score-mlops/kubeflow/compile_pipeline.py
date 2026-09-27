"""
kubeflow/compile_pipeline.py

Compiles kubeflow/pipeline.py into a static `pipeline.yaml` -- the portable, standard KFP
IR (intermediate representation) format that would be UPLOADED to a real Kubeflow Pipelines
cluster (via the KFP UI's "Upload Pipeline" button, or programmatically with
`kfp.Client().create_run_from_pipeline_package(...)`). Compiling does not need a cluster,
Docker, or even a `kfp.local` runner -- it's a pure, local, static translation of the
Python DAG definition into KFP's YAML pipeline spec.

WHY this file matters pedagogically: it's the bridge between the LOCAL teaching setup
(kubeflow/run_local.py -- what you actually run in class) and a PRODUCTION KUBEFLOW
ENVIRONMENT. The exact same `kubeflow/pipeline.py` definition produces both -- nothing
about the pipeline's LOGIC changes; only how/where it executes does. Running this script
and opening the resulting `pipeline.yaml` is a good way to actually SEE what a Kubeflow
pipeline "is", underneath the Python decorators: a plain-text, versionable spec of
components, their container images, their inputs/outputs, and the DAG/conditional
structure between them.

WHAT WOULD DIFFER in a real production deployment (this project's `base_image="python:3.11"`
components are correct but minimal -- a production team would typically):
    - build a proper Docker image per component (or a shared one) with this project's
      `requirements.txt` already baked in, rather than relying on `python:3.11` alone
      (these components only need the standard library + `subprocess`, which is why the
      bare image works here -- but a component that imported `pandas` etc. DIRECTLY, rather
      than shelling out to a script that has its own installed environment, would need
      `packages_to_install=[...]` or a custom base image).
    - point `repo_root` at a path INSIDE that image (or better: replace the "shell out to a
      script under repo_root" pattern with real `Input[Dataset]`/`Output[Model]` KFP
      artifacts backed by cloud object storage, so components don't need a shared
      filesystem at all -- Kubernetes Pods on different nodes don't have one).
    - run on a schedule via Kubeflow's own recurring-run feature (see README's "Scheduled
      Retraining" section) instead of a developer invoking this script by hand.

Run from the repo root:
    python -m kubeflow.compile_pipeline
    # -> writes kubeflow/pipeline.yaml
"""
from pathlib import Path

from kfp import compiler

from kubeflow.pipeline import exam_score_mlops_pipeline

OUTPUT_PATH = Path(__file__).resolve().parent / "pipeline.yaml"


def main():
    compiler.Compiler().compile(
        pipeline_func=exam_score_mlops_pipeline,
        package_path=str(OUTPUT_PATH),
    )
    print(f"Compiled pipeline -> {OUTPUT_PATH}")
    print("This file is what you would upload to a real Kubeflow Pipelines cluster's UI, "
          "or submit with kfp.Client().create_run_from_pipeline_package(...).")


if __name__ == "__main__":
    main()
