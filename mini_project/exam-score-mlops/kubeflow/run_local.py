"""
kubeflow/run_local.py

Runs the Kubeflow pipeline (kubeflow/pipeline.py) LOCALLY -- no Kubernetes cluster, no
Kubeflow deployment, no Docker required for the pipeline engine itself. This is the
"lightweight Kubeflow Pipelines setup" the course spec asks for: real `kfp` SDK code (real
`@dsl.component`s, a real `@dsl.pipeline`, real conditional DAG logic), executed by KFP's
own `kfp.local` runner instead of against a cluster.

WHAT `kfp.local` actually does: it's an OFFICIAL part of the `kfp` SDK (not a hand-rolled
workaround) that executes a compiled pipeline's components one at a time as local
subprocesses, following the exact same DAG/conditional logic a real KFP backend would use,
and writes each component's outputs under `local_outputs/` so you can inspect them after
the fact -- see that directory after running this script.

    local.init(runner=local.SubprocessRunner(use_venv=False))

`use_venv=False` means each component runs in the CURRENT Python environment (this
project's own venv/deps) rather than kfp building a fresh, isolated venv per component --
the right trade-off for a fast, iterative CLASSROOM demo where you already trust your own
environment. `local.SubprocessRunner(use_venv=True)` (or `local.DockerRunner`, which needs
a Docker daemon) gives stronger isolation between components, closer to production
behavior, at the cost of a slower first run per component -- worth showing students as the
next step up, without requiring it here.

LOCAL TEACHING SETUP (this script) vs. PRODUCTION KUBEFLOW ENVIRONMENT (see
kubeflow/compile_pipeline.py):
    | | Local (this script) | Production |
    |---|---|---|
    | Orchestrator | `kfp.local`, in-process | Kubeflow Pipelines backend on Kubernetes |
    | Where components run | local subprocess(es) | Kubernetes Pods, one per component |
    | Isolation between components | shared venv (or per-component venv/Docker) | full container isolation, every run |
    | Scale | your one machine | a cluster -- horizontally scalable |
    | Scheduling / retries / multi-user | none built in | Kubeflow's own scheduler, retry policies, RBAC |
    | Artifact storage | local disk (`repo_root`, `local_outputs/`) | object storage (S3/GCS/MinIO) via KFP's artifact store |
    | How you'd get there from here | -- | `kfp compile` (compile_pipeline.py) -> upload
      `pipeline.yaml` to a running KFP instance (e.g. via the KFP UI or `kfp.Client()`) --
      the PIPELINE DEFINITION itself does not need to change at all |

Run from the repo root:
    python -m kubeflow.run_local
    python -m kubeflow.run_local --run-docker-build
"""
import argparse
from pathlib import Path

from kfp import local

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run the Exam Score MLOps Kubeflow pipeline locally.")
    parser.add_argument("--run-docker-build", action="store_true",
                         help="If a candidate is promoted, also run a real `docker build` "
                              "(requires a local Docker daemon) -- otherwise the deployment "
                              "gate only prints the manual next steps.")
    args = parser.parse_args()

    # SubprocessRunner(use_venv=False): components run in THIS process's own environment.
    # local_outputs/ (under repo_root) is where each component's per-run logs/outputs land.
    local.init(
        runner=local.SubprocessRunner(use_venv=False),
        pipeline_root=str(REPO_ROOT / "local_outputs"),
    )

    # Imported AFTER local.init() -- kfp's local execution mode must be initialized before
    # a @dsl.pipeline-decorated function is actually CALLED (not merely defined).
    from kubeflow.pipeline import exam_score_mlops_pipeline

    print(f"\n{'=' * 74}\nRunning Kubeflow pipeline locally (repo_root={REPO_ROOT})\n{'=' * 74}\n")
    exam_score_mlops_pipeline(repo_root=str(REPO_ROOT), run_docker_build=args.run_docker_build)
    print(f"\nDone. Per-component logs/outputs: {REPO_ROOT / 'local_outputs'}")


if __name__ == "__main__":
    main()
