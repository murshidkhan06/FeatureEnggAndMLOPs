# exam_score_project_airflow

Orchestrating the **same exam_score_project** ML pipeline with **Apache Airflow** — one of
the tool deep-dives in the Feature Engineering & MLOps course (Semester VII), alongside the
companion DVC, MLflow, GitHub Actions, and Kubeflow modules on this same project.

## 1. Project overview

Predicts a student's final exam score from study hours, attendance %, mock test scores,
city, and income bracket — the exact same scikit-learn pipeline (`FeatureCreator` →
`ColumnTransformer` → `SelectFromModel(Lasso)` → `LinearRegression`) used throughout this
course. Nothing about the ML problem changes here; what changes is **how the pipeline's
steps are scheduled and run.**

## 2. Why Airflow, when this course already has Kubeflow?

They solve overlapping but genuinely different problems:

| | **Apache Airflow** | **Kubeflow Pipelines** |
|---|---|---|
| Core strength | General-purpose, **time-based** workflow scheduling | **ML-native**, Kubernetes-native pipeline execution |
| "Run this every Sunday at 2am, retry twice, alert on failure" | This is Airflow's home turf | Possible, but not the primary design point |
| Understands "model", "metric", "experiment" as first-class concepts | No — it schedules arbitrary Python/Bash tasks | Yes — artifacts and metrics are native outputs |
| Where it runs | Anywhere Python runs (a single VM is enough) | Built around running each step as its own container on a cluster |
| Also used for | Data warehouse ETL, report generation, literally any scheduled job | ML pipelines specifically |

**Neither replaces the other.** A real MLOps stack often has BOTH: Airflow scheduling the
*business* cadence ("retrain weekly," "refresh the data warehouse nightly"), and, for the
ML-specific steps, either plain Python (as here) or a Kubeflow/KFP pipeline as one of
Airflow's tasks.

## 3. Architecture

```
validate_data -> train_and_evaluate -> generate_new_batch -> check_drift
                                                                  |
                                                             decide_retrain
                                                          (@task.branch)
                                                     /                    \
                                            [drift detected]        [no drift]
                                                   v                        v
                                            retrain_model            skip_retrain
                                                   \                       /
                                                    v                     v
                                                     pipeline_summary
```

`decide_retrain` is Airflow's native conditional-branching mechanism
(`@task.branch` / `BranchPythonOperator`) — the direct analogue of Kubeflow's `dsl.If` in
the companion module. Airflow marks the branch NOT taken as **skipped** (not failed) in the
UI — a real, visible distinction worth showing students.

Every task is a thin subprocess wrapper around an already-tested CLI module
(`src/train.py`, `src/drift.py`, `src/retrain.py`) — the same "wrap, don't reimplement"
principle used throughout this course's Kubeflow module.

## 4. Two environments, on purpose

This is the single most important practical lesson in this module, and a real, common
production gotcha:

**Airflow's own Python environment does not need — and should not have — this project's ML
dependencies installed.** Airflow ships its own strict dependency pins (a
`constraints-<version>-<python>.txt` file); MLflow's newer `pydantic` / `typing-extensions`
/ `cryptography` requirements can conflict with those pins directly if installed into the
same environment. This project hit that conflict for real while being built (see
`docs/airflow_dag_test_transcript.txt`'s setup notes) and fixes it the way a real
deployment would:

- **`.venv/`** — the PROJECT environment: pandas, scikit-learn, mlflow, etc. This is what
  actually trains the model, evaluates it, checks drift, and retrains.
- **`.venv-airflow/`** — Airflow's OWN environment: just `apache-airflow`, nothing else.

The DAG (`airflow/dags/exam_score_training_dag.py`) explicitly points every subprocess call
at `.venv/bin/python`, never at whatever Python happens to be running Airflow itself. In a
real deployment, this same idea shows up as a dedicated task virtualenv, a
`PythonVirtualenvOperator`/`ExternalPythonOperator`, or — most commonly — each task running
in its own Docker/Kubernetes image.

## 5. Setup

```bash
cd exam_score_project_airflow

# 1. Project environment (the ML stack)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/raw/generate_data.py     # only if data/raw/student_exam_scores.csv is missing
deactivate

# 2. Airflow's OWN environment (kept separate on purpose -- see Sec. 4)
python -m venv .venv-airflow
source .venv-airflow/bin/activate
AIRFLOW_VERSION=2.10.3
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

## 6. Run the pipeline WITHOUT Airflow first (sanity check)

Exactly like the companion modules, every step is a plain CLI command you can run by hand:

```bash
source .venv/bin/activate
python -m src.train                                              # train + evaluate + log to MLflow
python -m src.generate_batch --n 150 --drift                     # simulate a new, drifted batch
python -m src.drift --current data/scenarios/new_batch.csv --numeric-only
python -m src.retrain --data-path data/scenarios/new_batch.csv   # only if drift.py found drift
```

## 7. Run the SAME pipeline as an Airflow DAG

**Fastest — no webserver, no scheduler, verifies the whole DAG in one command:**

```bash
source .venv-airflow/bin/activate
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow db migrate                                    # first time only
airflow dags test exam_score_training_pipeline 2026-01-01
```

Verified while building this project (`docs/airflow_dag_test_transcript.txt` is the real,
captured output): all 8 tasks ran, `decide_retrain` correctly detected drift in
`study_hours` and `attendance_pct`, branched into `retrain_model`, correctly **skipped**
`skip_retrain`, and the retrained candidate was legitimately **promoted** (trained on data
that actually reflected the drifted batch, evaluated against a champion that had never seen
it).

**With the full scheduler + webserver (the real classroom-demo experience):**

```bash
source .venv-airflow/bin/activate
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
airflow standalone     # starts scheduler + webserver, prints an admin password
# open http://localhost:8080, un-pause "exam_score_training_pipeline", trigger it manually
```

## 8. Scheduling

The DAG is defined with `schedule="@weekly"` — a real, standing cadence ("retrain every
week"), not "whenever someone remembers to run a script." Airflow's scheduler evaluates
this on its own once the webserver/scheduler are running; `airflow dags test` (above) always
runs immediately regardless of schedule, which is exactly why it's the right tool for a fast
classroom demo or CI check.

## 9. Common mistakes

| Mistake | Fix |
|---|---|
| Installing this project's ML dependencies into Airflow's own environment | Keep them separate (Sec. 4) — point subprocess calls at the project's own `.venv` |
| Assuming a `@task.branch` function can freely `import pandas` | That function runs inside AIRFLOW's environment when the scheduler parses/executes the DAG file — use the stdlib (`csv`, `json`) for anything read at that level |
| Auto-retraining the instant `decide_retrain` sees ANY drift | This DAG's branch is intentionally the simple, drift-only version for teaching Airflow's branching mechanism — a full production decision should also require confirmed performance decay and enough new data volume (see `src/retrain_decision.py` in the companion Monitoring & Kubeflow module) |
| Treating `airflow dags test` as equivalent to production scheduling | It runs the DAG immediately, ignoring `schedule` and skipping the scheduler/executor entirely — perfect for fast verification, not a substitute for actually running `airflow standalone` before trusting a schedule |

## 10. Project structure

```
exam_score_project_airflow/
├── airflow/
│   └── dags/
│       └── exam_score_training_dag.py   # the DAG: validate -> train -> drift -> branch -> retrain/skip
├── src/
│   ├── data_processing.py               # load + validate raw data
│   ├── features.py                      # FeatureCreator custom transformer
│   ├── pipeline_builder.py              # assembles the sklearn Pipeline
│   ├── train.py                         # train + evaluate + MLflow logging
│   ├── evaluate.py                      # metrics
│   ├── predict.py                       # load a saved pipeline, predict
│   ├── drift.py                         # KS test (numeric) + chi-square (categorical)
│   ├── retrain.py                       # candidate vs. champion promotion gate
│   └── generate_batch.py                # NEW: standalone "new batch" CSV, no API needed
├── api/main.py                          # FastAPI serving (same as the other modules)
├── data/
│   ├── raw/                             # original training data
│   ├── reference/                       # frozen training-time snapshot (drift baseline)
│   └── scenarios/                       # generated "new batch" CSVs (gitignored)
├── tests/                               # same test suite as the other modules
├── docs/
│   └── airflow_dag_test_transcript.txt  # real, captured `airflow dags test` output
├── params.yaml
├── requirements.txt                     # PROJECT environment (ML stack)
└── requirements-dev.txt                 # + pytest, etc.
```

Note: `apache-airflow` is deliberately **not** in `requirements.txt` — see Sec. 4. Install
it into its own `.venv-airflow/` per Sec. 5.

## 11. Curriculum mapping

| Component | Curriculum concept |
|---|---|
| `airflow/dags/exam_score_training_dag.py` | Workflow orchestration with a general-purpose scheduler |
| `@task.branch` / `decide_retrain` | Conditional DAG branching (compare to Kubeflow's `dsl.If`) |
| `schedule="@weekly"` | Scheduled vs. trigger-based execution |
| Two separate venvs (Sec. 4) | Dependency isolation between an orchestrator and its ML workloads |
| `default_args` (`retries`, `retry_delay`) | Reliability primitives a plain cron job doesn't give you for free |
| Sec. 2's comparison table | Airflow vs. Kubeflow vs. cron — choosing the right orchestrator for the job |
