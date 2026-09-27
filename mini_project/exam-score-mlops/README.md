# Student Exam Score Predictor — Complete Local MLOps Pipeline

A complete, locally-executable MLOps project built on top of the existing
`exam_score_project` teaching notebook. Same dataset, same feature engineering, same
Linear Regression model — this project wraps it with everything needed to go from a
notebook to a versioned, tracked, tested, containerized, monitored, and retrainable ML
system, entirely on your own machine. No cloud account required anywhere in the core
workflow.

## 1. Project overview

Predicts a student's final exam score from raw, in-progress signals (study hours,
attendance, three mock test scores, and basic profile data) so a coaching institute can
flag at-risk students early. The ML core — feature engineering, preprocessing, PCA,
feature selection, and a Linear Regression model, all inside one leakage-safe
`sklearn.Pipeline` — is reused unchanged from `mini_project/exam_score_project/`. What's
new here is everything around it: Git + DVC versioning, MLflow tracking and a model
registry, automated tests, CI, Docker packaging, observability, drift detection, and a
retrain-and-promote loop.

## 2. Business problem

A coaching institute wants to predict a student's **final exam score** from data
collected *during* the course — study habits, attendance, three mock test scores, and
basic profile info — so they can intervene early for students who need it, not after the
final exam.

## 3. Architecture

```
                 Git
                  |
                  v
              Source Code
                  |
                  v
Raw Data --> DVC --> Training Pipeline (src/pipeline.py)
                  |
                  v
          Feature Engineering (src/features.py, reused from the notebook)
                  |
                  v
               Model (src/pipeline_builder.py: FeatureCreator -> preprocess -> PCA
                  |     -> Lasso feature selection -> LinearRegression)
                  v
              MLflow (sqlite:///mlflow.db)
          /       |       \
     Params    Metrics   Artifact
                           |
                           v
                     Model Registry (alias: "champion")
                           |
                           v
                       Packaging (joblib)
                           |
                           v
                       Docker
                           |
                     Docker Hub
                           |
                           v
                       FastAPI (api/main.py)
                           |
                           v
                      Prediction
                           |
                    +------+------+
                    v             v
               Monitoring      Logs
           (monitoring/predictions.csv, app.log)
                    |
                    v
     +----------------------------------------------------------+
     |  UNIT 4 (see "Unit 4 Extension" section below for the     |
     |  full story -- WHY/WHAT/HOW for every piece)              |
     |                                                            |
     |  Delayed labels --(POST /feedback)--> monitoring/labels.csv
     |         |                                                  |
     |         v                                                  |
     |  Data Drift (src/drift.py: numeric KS test + categorical   |
     |  chi-square)      Performance Decay (src/performance_      |
     |         |          monitor.py: needs delayed labels)       |
     |         +---------------------+                            |
     |                               v                            |
     |                        Alerting (src/alerts.py)            |
     |                               |                            |
     |                               v                            |
     |        Retrain Trigger Decision (src/retrain_decision.py   |
     |        -- drift alone is NOT enough; needs CONFIRMED       |
     |        performance decay)                                  |
     |                               |                            |
     |                               v                            |
     |   Kubeflow Pipeline (kubeflow/) orchestrates:               |
     |     Validate -> Train+Eval+MLflow -> Drift Check ->         |
     |     Trigger Decision -> [if RETRAIN] Retrain + Promote/     |
     |     Reject -> [if PROMOTED] Deploy Gate                     |
     +----------------------------------------------------------+
                    |
                    v
                Retraining (src/retrain.py: champion vs. candidate, promotion gate)
                    |
                    +----> back to Training Pipeline, IF promoted
```

**Training pipeline** (`src/pipeline.py`) and **application deployment** (Docker) are
kept deliberately decoupled — retraining a model never automatically rebuilds or
redeploys a container, and vice versa. See Phase mapping in section 28.

## 4. Technology stack

| Layer | Tool | Why |
|---|---|---|
| ML core | pandas, numpy, scikit-learn | reused unchanged from the notebook |
| Config | `params.yaml` | one place for every hyperparameter, read by DVC and MLflow both |
| Data versioning | DVC (local, no remote) | Git isn't built for large/binary data files |
| Experiment tracking + registry | MLflow (`sqlite:///mlflow.db`) | plain file-store MLflow doesn't support the Model Registry; SQLite does, and stays 100% local |
| Packaging | joblib | see section 7 — same reasoning as the original notebook |
| API | FastAPI + Pydantic + uvicorn | reused from `app/main.py`, with observability added |
| Testing | pytest | data / feature / model / API test layers |
| CI/CD | GitHub Actions | test on every push; Docker build validated, push left manual |
| Containerization | Docker + docker-compose | independently runnable, no host Python needed |
| Drift detection | `scipy.stats.ks_2samp` (numeric), `scipy.stats.chi2_contingency` (categorical) | distribution-free two-sample test; frequency-table test for labels with no order/distance |
| Performance monitoring | `sklearn.metrics` on delayed labels | needs actual outcomes, not just inputs -- see "Unit 4 Extension" |
| Alerting | `src/alerts.py` (threshold-based) | print + log + report file locally; swappable for Slack/PagerDuty in production |
| Workflow orchestration | **Kubeflow Pipelines (`kfp` SDK, `kfp.local` runner)** | a DAG-based orchestrator with conditional branches -- see "Unit 4 Extension" |

`python -m src.pipeline` / `make pipeline` remains the orchestrator for the TRAINING
pipeline alone (Units 1-3). The **full production lifecycle** -- validate, train,
drift-check, decide whether to retrain, retrain, promote, and gate deployment -- is
orchestrated by a real Kubeflow Pipeline (`kubeflow/`), runnable **locally with no
Kubernetes cluster** via `kfp.local`. See "Unit 4 Extension" below.

## 5. Project structure

```
exam-score-mlops/
├── data/{raw,processed,reference,scenarios}/   # scenarios/ generated by Unit 4 demos
├── notebooks/
│   ├── exam_score_pipeline.ipynb                          # Units 1-3, kept for lineage
│   └── unit4_monitoring_drift_retraining_kubeflow.ipynb    # Unit 4
├── docs/unit4_classroom_demo_transcript.txt    # a real, captured run of the demo below
├── scripts/classroom_demo.py                   # the 20-step / 5-scenario end-to-end demo
├── src/
│   ├── features.py            # FeatureCreator -- reused verbatim from the notebook
│   ├── pipeline_builder.py    # builds the ONE sklearn Pipeline, params.yaml-driven
│   ├── data_processing.py     # load, validate, split
│   ├── train.py                # python -m src.train
│   ├── evaluate.py
│   ├── predict.py              # shared load/predict helpers
│   ├── batch_predict.py        # python -m src.batch_predict
│   ├── monitoring.py           # logging + prediction log + /metrics + labels.csv (Unit 4)
│   ├── generate_traffic.py     # simulate real / drifted traffic against a running API
│   ├── drift.py                 # python -m src.drift (numeric KS + categorical chi-square)
│   ├── concept_drift_demo.py    # python -m src.concept_drift_demo (Unit 4)
│   ├── performance_monitor.py   # python -m src.performance_monitor (Unit 4)
│   ├── alerts.py                # python -m src.alerts (Unit 4)
│   ├── retrain_decision.py      # python -m src.retrain_decision (Unit 4)
│   ├── retrain.py              # python -m src.retrain
│   └── pipeline.py             # python -m src.pipeline (Unit 1-3 training-only orchestration)
├── kubeflow/                    # Unit 4: the full lifecycle as a Kubeflow Pipeline
│   ├── components.py            # @dsl.component wrappers around the src/ modules above
│   ├── pipeline.py               # @dsl.pipeline -- the DAG, with conditional branches
│   ├── run_local.py              # python -m kubeflow.run_local (kfp.local, no cluster)
│   └── compile_pipeline.py       # python -m kubeflow.compile_pipeline -> pipeline.yaml
├── api/main.py                 # FastAPI app (+ /health/live, /health/ready, /feedback -- Unit 4)
├── tests/{test_data,test_features,test_model,test_api}.py
├── models/exam_score_pipeline.joblib
├── monitoring/{predictions.csv, labels.csv, app.log}     # generated at runtime
├── reports/                    # generated at runtime (evaluation, drift, retrain reports)
├── .dvc/  .github/workflows/ml-pipeline.yml
├── Dockerfile  docker-compose.yml  .dockerignore
├── requirements.txt  requirements-dev.txt
├── dvc.yaml  dvc.lock  params.yaml  Makefile
└── README.md
```

## 6-9. Setup

**Prerequisites:** Python 3.11+ (this project was built and tested against 3.11), Docker
Desktop (for the Docker sections), Git.

```bash
git clone <this-repo>   # or just cd into it if you already have it locally
cd exam-score-mlops

python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```
Linux/macOS:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements-dev.txt   # everything, including test + DVC tooling
# or, for a serving-only install (what the Docker image actually uses):
pip install -r requirements.txt
```

## 10. Run the notebook

The notebook is kept for teaching lineage — it's the original exploration and explanation
of every feature engineering decision. It is **not** part of the production training path
(`src/train.py` is); running it is optional:

```bash
jupyter notebook notebooks/exam_score_pipeline.ipynb
```

## 11. Run training

```bash
python data/raw/generate_data.py    # regenerate the dataset (fixed seed -- deterministic)
python -m src.train
```

This validates the data, builds and fits the pipeline, evaluates it, logs everything to
MLflow, registers the model, saves `models/exam_score_pipeline.joblib`, and writes
`reports/evaluation_report.json` + `reports/feature_coefficients.csv`. Expect
**R²≈0.81, MAE≈3.9** — matching the original notebook exactly, since the pipeline logic is
unchanged.

## 12. Run MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000`. You'll see the `exam_score_prediction` experiment, with each
run's parameters (Lasso alpha, test size, etc.), metrics (R², MAE, RMSE), and artifacts
(the model, the coefficients CSV, the evaluation report). Under **Models**, you'll find
`exam_score_predictor` with one version per training run — this is only possible locally
because `params.yaml`'s `mlflow.tracking_uri` points at a SQLite file, not the plain
file-store MLflow defaults to.

## 13. Run the tests

```bash
pytest tests/ -v
```

39 tests across four files, verified passing end-to-end while building this project:

- **`test_data.py`** (9) — required columns exist, target has no missing values and is
  in-range, `study_hours` is *allowed* to be missing (that's by design), validation
  rejects malformed data.
- **`test_features.py`** (7) — `FeatureCreator` produces the right columns, doesn't mutate
  its input, and — importantly — each row's engineered features depend only on that row
  (no cross-row leakage).
- **`test_model.py`** (10) — the pipeline can be built and trained from scratch and clears
  a real MAE performance floor (not just "doesn't crash"), predictions are finite and
  correctly shaped, deterministic, directionally sane, and robust to missing values /
  unseen categories.
- **`test_api.py`** (13) — health/metrics endpoints, valid predictions, schema shape,
  API-matches-direct-pipeline-call, graceful handling of an unseen city, and proper 422s
  for missing fields, invalid categories, out-of-range values, and wrong types.

## 14. Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
    "study_hours": 12.5, "attendance_pct": 88.0,
    "mock_test_1": 72.0, "mock_test_2": 75.0, "mock_test_3": 70.0,
    "income_bracket": "Medium", "city": "Pune",
    "enrollment_date": "2025-06-01", "shoe_size": 9.0, "lucky_number": 42
}'
# -> {"predicted_final_score": 95.2, "model_version": "...", "request_id": "..."}

curl http://localhost:8000/metrics
```

Or open `http://localhost:8000/docs` for the interactive Swagger UI, generated
automatically from the same Pydantic schema that validates real requests.

Every successful or failed prediction is appended to `monitoring/predictions.csv` and
logged to `monitoring/app.log` — see section 21.

## 15. Batch inference

```bash
python -m src.batch_predict --input data/raw/student_exam_scores.csv --output reports/batch_predictions.csv
```

Scores every row of a CSV using the *same* saved pipeline the API uses — no separate
batch-preprocessing logic to keep in sync. Prefer batch over real-time when you need many
predictions at once and nobody is waiting on an individual HTTP response (e.g. "score every
enrolled student tonight"); prefer real-time when a single prediction needs to come back
immediately in response to a user action.

## 16. Build the Docker image

```bash
docker build -t <dockerhub-username>/exam-score-api:1.0 .
```

Replace `<dockerhub-username>` with your own Docker Hub username (or drop it entirely to
just tag it `exam-score-api:1.0` locally). **The model must be trained first** —
`python -m src.train` — since the Dockerfile copies `models/` into the image.

## 17. Run Docker

```bash
docker run -p 8000:8000 <dockerhub-username>/exam-score-api:1.0
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
```

The container is independently runnable — it does not depend on your host's Python
environment at all.

Or run the API + MLflow UI together:
```bash
docker compose up --build
```

## 18. Docker Hub — push

```bash
docker login
docker build -t <dockerhub-username>/exam-score-api:1.0 .
docker tag <dockerhub-username>/exam-score-api:1.0 <dockerhub-username>/exam-score-api:latest
docker push <dockerhub-username>/exam-score-api:1.0
docker push <dockerhub-username>/exam-score-api:latest
```

## 19. Docker Hub — pull

```bash
docker rmi <dockerhub-username>/exam-score-api:1.0        # remove the local image
docker pull <dockerhub-username>/exam-score-api:1.0        # pull it back from Docker Hub
docker run -p 8000:8000 <dockerhub-username>/exam-score-api:1.0
curl http://localhost:8000/health
```

`build -> tag -> push -> Docker Hub -> pull -> run -> FastAPI -> prediction` — a working
image at `<dockerhub-username>/exam-score-api` is now something ANYONE can run, without
your source code, your Python environment, or your trained model file individually.

## 20. Git workflow

```bash
git init                       # already done if you're reading this from a clone
git add .
git commit -m "initial commit: MLOps project scaffolded from exam_score_project"

git checkout -b dev
# ... make changes ...
git add <files>
git commit -m "meaningful, specific message"
git checkout main
git merge dev

git log --oneline
git tag v1.0 -m "first trained + tested + containerized model"
```

Suggested branching: `main` (stable, always deployable) / `dev` (integration) /
`feature/*` (one change at a time). **Code versioning** (Git) tracks *what logic produced
a model*; **dataset versioning** (DVC, next section) tracks *what data trained it*; **model
versioning** (MLflow Registry) tracks *which trained artifact is currently in production*
— three different questions, three different tools, used together.

## 21. DVC workflow

```bash
dvc init                                        # already done in this repo
dvc add data/raw/student_exam_scores.csv        # creates student_exam_scores.csv.dvc
git add data/raw/student_exam_scores.csv.dvc .gitignore
git commit -m "track dataset v1 with DVC"

dvc repro                                        # reproduce the train stage (dvc.yaml) --
                                                  # skips retraining if nothing changed
```

**Change the dataset and create a new version:**
```bash
python data/raw/generate_data.py   # or edit the CSV directly
dvc add data/raw/student_exam_scores.csv
git add data/raw/student_exam_scores.csv.dvc
git commit -m "dataset v2: ..."
```

**Switch back to an earlier data version:**
```bash
git checkout <earlier-commit-hash> -- data/raw/student_exam_scores.csv.dvc
dvc checkout
```

This project uses **local-only** DVC storage (no cloud dependency) — the `.dvc/cache`
directory on your own machine is where DVC actually keeps each data version. To add a
real remote later (so a teammate can `dvc pull` without your machine being reachable):

```bash
dvc remote add -d myremote /path/to/some/other/local/or/network/folder
# or, e.g.: dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push
```

## 22. CI/CD

`.github/workflows/ml-pipeline.yml` runs on every push/PR to `main`/`dev`:

- **CI** (`test` job, always runs): installs deps, regenerates the dataset, validates it,
  trains the model, runs the full test suite, and import-checks the API module.
- **CD** (`build-docker-image` job, push only): builds the Docker image as a
  *validation* step. It does **not** push to Docker Hub by default — that requires
  `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` repo secrets, which this template deliberately
  leaves commented out rather than assuming you've configured them.

**CI** = automatically test code/data/model changes. **CD** = automatically
build/package/deploy the application. Keeping them as separate jobs (and CD as an
opt-in, secrets-gated step) keeps the default pipeline fast, free, and safe to fork.

## 23. Monitoring

```bash
python -m src.monitoring
```

Summarizes `monitoring/predictions.csv` — total predictions, success/error counts,
average latency, average predicted score. The running API also exposes live counters at
`GET /metrics`. Generate some realistic traffic first:

```bash
uvicorn api.main:app --port 8000     # in one terminal
python -m src.generate_traffic --n 200      # in another
```

## 24. Drift detection

```bash
python -m src.drift
```

Runs a KS test comparing `data/reference/training_reference.csv` (a frozen snapshot of the
raw training inputs, written automatically by `src/train.py`) against whatever's currently
in `monitoring/predictions.csv`. Verified while building this project:

- 150 requests sampled from the *same* distribution as training data → **no drift
  detected** on any of the 7 numeric features.
- 150 requests with `study_hours` and `attendance_pct` deliberately halved/reduced
  (`python -m src.generate_traffic --n 200 --simulate-drift`) → **drift correctly
  detected** on exactly those two features, and *only* those two.

**Statistical significance is not business significance** — a p-value below the threshold
tells you the distributions look different; it does not by itself tell you the model's
real-world accuracy has degraded. Read the KS statistic and the actual data next to the
p-value before treating a drift alert as an action item.

## 25. Retraining

```bash
python -m src.retrain
# or, to retrain on a different/updated CSV:
python -m src.retrain --data-path path/to/new_data.csv
```

Trains a **candidate** model, evaluates it on the same held-out split as the current
**champion** (the live `models/exam_score_pipeline.joblib`), and promotes the candidate
**only if**:

```
candidate_mae <= champion_mae - min_mae_improvement   (params.yaml: retraining.min_mae_improvement)
AND
candidate_mae <= max_acceptable_mae                    (params.yaml: evaluation.max_acceptable_mae)
```

Drift detected ≠ the model needs replacing — retraining on the same data as before
correctly produces a candidate that is **rejected** (no improvement), which is exactly
what you want a promotion gate to do. Promoted candidates get the MLflow Registry alias
`champion`; rejected ones get `rejected_candidate`, kept for audit rather than discarded.

**Scheduled vs. triggered retraining:** for the local project, retraining is a manual
command (above). To schedule it, either a plain OS **cron** entry (`0 2 * * 0` for weekly)
calling `python -m src.retrain`, or a GitHub Actions
[`schedule`](https://docs.github.com/actions/using-workflows/events-that-trigger-workflows#schedule)
trigger added to a copy of the CI workflow. **Trigger-based** retraining would call the
same command from `src/drift.py`'s alert path once drift is confirmed — deliberately not
wired up automatically in this template, since "drift detected" should prompt a human
decision, not an unattended retrain-and-deploy.

## 26. Orchestration

```bash
python -m src.pipeline
# or
make pipeline
```

Runs data validation → train → evaluate → MLflow logging → model artifact → test suite,
as one command. Deliberately stops there — Docker build/deploy is a separate, manual step
(`make docker-build`), keeping the **ML training pipeline** and the **application
deployment pipeline** decoupled, so retraining never accidentally redeploys a container
and packaging never accidentally retrains a model.

## 27. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `FileNotFoundError: Raw data not found` | Run `python data/raw/generate_data.py`, or `dvc checkout` if you have a tracked version |
| `RuntimeError: Model file not found` (API won't start) | Run `python -m src.train` first |
| `AttributeError: Can't get attribute 'FeatureCreator'` | Something is loading the joblib file without importing `src.features` first — see `src/features.py`'s docstring |
| `pytest` skips most tests | No trained model / dataset yet — see the two rows above |
| `docker build` fails at `FROM python:3.11-slim` | Docker Desktop isn't running, or you have no internet access to Docker Hub at that moment |
| `docker run` fails with "port is already allocated" | Another process is already using port 8000 — stop it, or run with `-p 8001:8000` and adjust your curl calls |
| MLflow UI shows no experiments | Check you're pointing `--backend-store-uri` at the SAME `sqlite:///mlflow.db` file training wrote to (must run `mlflow ui` from the repo root) |
| `dvc repro` says "didn't change, skipping" every time | That's correct behavior once nothing (data/code/params) has changed since the last run — edit `params.yaml` or the data to see it actually retrain |
| Drift report shows drift on EVERY feature, even for normal traffic | Very small sample size makes the KS test unstable — send more traffic first (`python -m src.generate_traffic --n 150+`) |

## 28. Learning outcomes — curriculum mapping

| Component | Curriculum concept |
|---|---|
| `src/features.py`, `src/pipeline_builder.py` | Feature Engineering review (Unit 3 §1) — reused, not redesigned |
| Git branches/commits/tags | Code versioning |
| DVC | Dataset versioning + lineage |
| `params.yaml` + `dvc.yaml`/`dvc.lock` | Reproducibility |
| MLflow tracking | Experiment tracking |
| MLflow Model Registry (`champion` alias) | Model Registry vs. experiment tracking |
| `joblib` (`src/train.py`) | Model packaging; pickle-vs-joblib and serialization risk (see §29) |
| `tests/` (pytest) | ML testing across data / feature / model / API layers |
| `.github/workflows/ml-pipeline.yml` | CI/CD for ML |
| `api/main.py` (FastAPI) | Model serving / deployment |
| `src/batch_predict.py` | Batch vs. real-time inference |
| `Dockerfile`, `docker-compose.yml` | Deployment / containerization |
| Docker Hub push/pull | Container registries |
| `src/monitoring.py`, `monitoring/predictions.csv` | Observability, prediction logging |
| `src/drift.py` (KS test) | Data drift detection |
| `src/retrain.py`'s champion/candidate comparison | Model drift / performance-decay monitoring concept |
| `src/retrain.py`'s promotion gate | Automated retraining with a safety gate |
| `src/pipeline.py` / `make pipeline` | End-to-end ML pipeline orchestration (Units 1-3, training only) |
| `src/drift.py`'s `run_chi_square_drift_check` | Categorical/discrete-feature drift detection |
| `src/concept_drift_demo.py` | Data drift ($P(X)$) vs. concept drift ($P(Y\|X)$) — the distinction, made concrete |
| `api/main.py`'s `GET /health/live` vs. `GET /health/ready` | Observability — liveness vs. readiness semantics |
| `api/main.py`'s `POST /feedback`, `src/monitoring.record_actual_label` | Delayed labels / prediction store pattern |
| `src/performance_monitor.py` | Model performance-decay monitoring (needs real outcome labels, not just drift) |
| `src/alerts.py` | Threshold-based alerting, single swappable delivery point |
| `src/retrain_decision.py` | Trigger-based retraining decision logic — "drift detected ≠ retrain" |
| `kubeflow/components.py`, `kubeflow/pipeline.py` | Kubeflow Pipelines — components, parameters, conditional (`dsl.If`) DAGs |
| `kubeflow/run_local.py` (`kfp.local`) vs. `kubeflow/compile_pipeline.py` | Local teaching execution vs. the portable, cluster-deployable pipeline definition |
| `scripts/classroom_demo.py`, `docs/unit4_classroom_demo_transcript.txt` | Full end-to-end lifecycle + the five required retraining/promotion scenarios |

## 29. Production considerations

This project is a **local teaching exercise**; a few things would be handled differently
in a real production deployment:

- **Trust your pickles.** `joblib`/`pickle` can execute arbitrary code on load — never
  load a `.joblib`/`.pkl` file from an untrusted source. `src/train.py` deliberately
  overrides MLflow's default `skops` serializer to demonstrate this exact trade-off; a
  real production system might instead invest in `skops`-compatible custom transformers
  to get its safety guarantees.
- **No secrets in Git.** Docker Hub credentials, cloud storage keys, etc. belong in
  environment variables or a secrets manager (GitHub Actions repo secrets, for the CI
  workflow here) — never committed.
- **Don't log sensitive data.** `src/monitoring.py` logs only fields already part of the
  model's own input schema, nothing incidental.
- **Validate all API inputs.** Already true here via Pydantic — never relax this to "trust
  the caller."
- **Pin dependencies.** `requirements.txt` is version-pinned for exactly this reason (see
  section 4).
- **Use an authenticated model registry** in a real deployment (MLflow behind auth, or a
  managed registry) — this project's local SQLite-backed registry has no access control.
- **Scan Docker images** for known vulnerabilities before deploying (e.g. `docker scout`,
  Trivy) — not done automatically here.
- **Real production monitoring** needs proper observability infrastructure (Prometheus/
  Grafana, structured log aggregation, alerting integrations) — this project's
  CSV-and-plain-log approach is intentionally the simplest thing that could work locally,
  not a production monitoring stack. `GET /metrics` returning plain JSON is a natural
  starting point for wiring up a real Prometheus exporter later.

## 30. Day 1 → Production: full walkthrough

```bash
# 1. Clone / open the repo
cd exam-score-mlops

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install requirements
pip install -r requirements-dev.txt

# 4. Get / version the dataset with DVC
python data/raw/generate_data.py
dvc init          # already done in this repo -- shown for a fresh clone
dvc add data/raw/student_exam_scores.csv

# 5. Run training
python -m src.train

# 6. Open MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # -> http://localhost:5000

# 7. Run tests
pytest tests/ -v

# 8. Start FastAPI
uvicorn api.main:app --port 8000

# 9. Call /predict
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'

# 10. Build Docker image
docker build -t <dockerhub-username>/exam-score-api:1.0 .

# 11. Run Docker locally
docker run -p 8000:8000 <dockerhub-username>/exam-score-api:1.0

# 12. Test API inside container
curl http://localhost:8000/health

# 13. Push image to Docker Hub
docker login && docker push <dockerhub-username>/exam-score-api:1.0

# 14. Remove local image
docker rmi <dockerhub-username>/exam-score-api:1.0

# 15. Pull image from Docker Hub
docker pull <dockerhub-username>/exam-score-api:1.0

# 16. Run pulled image
docker run -p 8000:8000 <dockerhub-username>/exam-score-api:1.0

# 17. Generate prediction traffic
python -m src.generate_traffic --n 200

# 18. Run monitoring
python -m src.monitoring

# 19. Run drift detection
python -m src.drift
# to see a detected-drift example:
python -m src.generate_traffic --n 150 --simulate-drift && python -m src.drift

# 20. Trigger retraining
python -m src.retrain

# 21. Evaluate new model
# -- printed automatically by src/retrain.py; also see reports/retrain_report.json

# 22. Promote new model if it passes quality criteria
# -- automatic: src/retrain.py only overwrites models/exam_score_pipeline.joblib and sets
#    the MLflow "champion" alias when the candidate actually clears the promotion gate
#    (see section 25). Nothing to do manually here unless you WANT to override the gate.
```

Every step above was run end-to-end while building this project (§13's 39 tests passing,
§24's drift-detected-only-on-the-shifted-features result, §25's correctly-rejected
identical-data retrain) — this is a verified, not theoretical, walkthrough.

## 31. Unit 4 Extension — Monitoring, Drift, Retraining & Kubeflow

Unit 4 does **not** create a new ML problem. It extends this SAME Exam Score project with
everything needed to operate it safely in production: real-time drift detection, delayed-label
performance monitoring, threshold-based alerting, a defensible retraining decision, and a
Kubeflow-orchestrated end-to-end pipeline. Every module below is real, executable, tested code
— not an architecture diagram standing in for code that doesn't exist.

### 31.1 What already existed (Units 1-3) → what we keep → what we extend → what we add

| | Component | Decision |
|---|---|---|
| **KEEP, unchanged** | `src/features.py`, `src/pipeline_builder.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`, `api/main.py`'s `/predict`, MLflow tracking/registry, DVC, Docker, CI | These already work and Unit 4 must not break them (all 39 pre-existing tests still pass) |
| **EXTEND** | `src/drift.py` | Add categorical (chi-square) drift alongside the existing numeric KS drift |
| **EXTEND** | `src/monitoring.py` | Add a second log (`monitoring/labels.csv`) for delayed ground-truth outcomes |
| **EXTEND** | `api/main.py` | Add `POST /feedback`; split `GET /health` into `GET /health/live` / `GET /health/ready` (kept `/health` for backward compatibility) |
| **EXTEND** | `params.yaml` | Add `drift.*`, `performance.*`, `alerting.*`, `retraining.*` sections — one source of truth for every threshold |
| **ADD, new** | `src/concept_drift_demo.py` | A honest, provable data-drift-vs-concept-drift demonstration |
| **ADD, new** | `src/performance_monitor.py` | Joins predictions ⋈ labels, compares recent performance against the training-time baseline |
| **ADD, new** | `src/alerts.py` | Threshold-based alerting with one swappable delivery function |
| **ADD, new** | `src/retrain_decision.py` | The "drift ≠ retrain" decision table, with an `--execute` flag |
| **ADD, new** | `kubeflow/` (`components.py`, `pipeline.py`, `run_local.py`, `compile_pipeline.py`) | A real, locally-runnable, conditionally-branching Kubeflow Pipeline that orchestrates all of the above |
| **ADD, new** | `scripts/classroom_demo.py`, `docs/unit4_classroom_demo_transcript.txt` | A full, runnable, 20-step lifecycle demo covering all five required teaching scenarios |
| **ADD, new** | `notebooks/unit4_monitoring_drift_retraining_kubeflow.ipynb` | The classroom-facing teaching notebook for everything above (WHY → WHAT → HOW → MATH → CODE → INDUSTRY → MISTAKES → PRODUCTION, per topic) |

### 31.2 Command reference

```bash
# Categorical + numeric drift (both by default)
python -m src.drift                      # numeric (KS) + categorical (chi-square)
python -m src.drift --numeric-only
python -m src.drift --categorical-only

# Concept drift demonstration (P(X) unchanged, P(Y|X) changed)
python -m src.concept_drift_demo

# Record a delayed ground-truth label for a past prediction
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "<id from a /predict response>", "actual_final_score": 78.5}'

# Liveness / readiness (Kubernetes-style)
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# Performance-decay monitoring (needs delayed labels via /feedback first)
python -m src.performance_monitor

# Threshold-based alerting (drift + performance + API error rate)
python -m src.alerts

# Trigger-based retraining decision (does NOT retrain by itself)
python -m src.retrain_decision
python -m src.retrain_decision --execute   # only actually retrains if the decision is RETRAIN

# Kubeflow: compile the portable pipeline spec (no cluster needed)
python -m kubeflow.compile_pipeline        # writes kubeflow/pipeline.yaml

# Kubeflow: run the full conditional DAG locally (no Kubernetes, no Docker required)
python -m kubeflow.run_local

# Full 20-step lifecycle + all five scenarios, in one script
python scripts/classroom_demo.py
```

### 31.3 The central lesson, demonstrated five ways

`scripts/classroom_demo.py` proves the single most important judgment call in this unit with
five real, executable scenarios (full transcript: `docs/unit4_classroom_demo_transcript.txt`):

| # | Setup | Verified outcome | What it proves |
|---|---|---|---|
| 1 | Normal traffic, no drift | `NO_RETRAIN` | A healthy system correctly does nothing |
| 2 | Data drift, but labels show performance still fine | `NO_RETRAIN` | **Drift alone never triggers a retrain** |
| 3 | Data drift + confirmed performance decay | `RETRAIN` | Retraining needs real evidence of harm, not just a shifted input distribution |
| 4 | Retrain on the SAME old data the champion already knows | `REJECTED` | A retrain with no new information cannot legitimately clear the promotion bar |
| 5 | Retrain on data reflecting the NEW (post-drift) reality | `PROMOTED` | Retraining only helps when given data that actually reflects what changed |

> **DRIFT DETECTED ≠ AUTOMATIC RETRAINING.**

### 31.4 Kubeflow: local teaching setup vs. a real production environment

| | This project (local) | Production |
|---|---|---|
| Orchestrator | `kfp.local`, in-process, `SubprocessRunner` | Kubeflow Pipelines backend on Kubernetes |
| Where components run | local subprocesses | one Kubernetes Pod per component |
| Artifact storage | local disk (`reports/`, `models/`) | object storage (S3/GCS/MinIO) |
| Scheduling, retries, multi-user | none built in | Kubeflow's own scheduler / retry policy / RBAC |
| Getting from here to there | — | `python -m kubeflow.compile_pipeline` produces the exact same `pipeline.yaml` IR a real cluster's UI accepts — the pipeline **definition** in `kubeflow/pipeline.py` does not change |

Kubeflow orchestrates the workflow; it does not replace MLflow (which tracks what happened
inside a training run), DVC (dataset versioning), Docker (packaging), or FastAPI (serving) —
see the full comparison table in the teaching notebook, §9.

**A real, documented `kfp.local` v2.17.0 limitation, worth knowing if you extend this
project:** a `dsl.If` condition that reads a *named key* of a multi-output
`NamedTuple` component's output (`task.outputs["key"]`) silently fails to enter the branch
(pipeline still reports overall `SUCCESS`, with only a benign-looking
`Could not resolve parent input` warning). The fix, applied throughout
`kubeflow/components.py` / `kubeflow/pipeline.py`: any component whose output drives a
`dsl.If` returns a single plain `str`, referenced via `.output` — documented in both files
so a future maintainer doesn't reintroduce it.

### 31.5 Where to learn this

- **Concepts, math, and worked examples:** `notebooks/unit4_monitoring_drift_retraining_kubeflow.ipynb` (verified to execute cleanly end-to-end via `jupyter nbconvert --execute`)
- **See it all run for real:** `python scripts/classroom_demo.py`, or read `docs/unit4_classroom_demo_transcript.txt`
- **Exercises:** end of the notebook (§11) — threshold sensitivity, building your own concept-drift scenario, extending alerting, adding a Kubeflow component, liveness vs. readiness reasoning
