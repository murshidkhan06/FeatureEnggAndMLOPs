# Credit Scoring - a regulated-lending ML system you can run on your laptop (Project 1)

A retail bank must decide, **at the moment a loan application is submitted**, whether to approve an unsecured personal loan.
The model returns the **probability of default within 12 months (PD)**. A human underwriter reviews the MEDIUM band.

This repo is a small but *real-looking* ML system: synthetic data -> business features -> feature selection with governance ->
leakage-safe training -> MLflow -> explainable API (reason codes) -> fairness audit + model review pack + GO/NO-GO gate ->
Docker / Kubernetes -> monitoring (Prometheus, Grafana, KS drift) -> **retraining with human approval and rollback**.

```
DATA -> FEATURE ENGINEERING -> FEATURE SELECTION + GOVERNANCE -> TRAINING -> EXPERIMENT TRACKING -> REVIEW GATE -> DEPLOYMENT (API)
     -> MONITORING -> DRIFT DETECTION -> (evidence + human decision) -> RETRAINING -> champion/challenger -> DEPLOYMENT ...
```

What this project teaches that the other projects do not: **regulation**. A credit model must be explainable (reason codes), fair
(adverse impact ratio), auditable (feature registry, model review pack) and must never use information from the future or sensitive attributes.
Technical accuracy is only one of several gates.

> All data is synthetic (seeded, see `docs/DATASET.md`). Numbers are plausible for teaching, not facts about any real bank or population.
> Laws about age, location, gender and explainability differ by country: the *policy decisions* in this repo are examples, not legal advice.

---------------------------------------------------------------------------------------------------------------------------------------

## 1. Quick start (about 5 minutes)

**Windows PowerShell** (Python 3.11 recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data.make_dataset                   # 1. generate data (seconds)
python -m src.features.build_features             # 2. time-based split + reference sample
python -m src.features.select_features            # 3. choose features on TRAINING rows only (filter / wrapper / embedded + governance)
python -m src.models.train --bootstrap-champion   # 4. baseline vs scorecard vs boosting, log to MLflow, install champion v1
pytest -q                                         # 5. the test suite
python scripts/quality_gate.py                    # 6. CI-style minimum metrics
uvicorn src.api.app:app --port 8000               # 7. serve (Swagger UI: http://localhost:8000/docs)
```
(or simply `.\scripts\run_local.ps1`)

**Linux / macOS / WSL** - the same commands, activate with `source .venv/bin/activate` (or run `bash scripts/run_local.sh`).

Then, in a second terminal:
```
python scripts/smoke_test.py --url http://localhost:8000          # health + one valid + one invalid request
python scripts/send_traffic.py --csv data/production/batch_normal.csv --n 200
python scripts/api_session.py --url http://localhost:8000         # the full API demo (valid, invalid, rejected sensitive field, latency)
```
Always run python **from the project root as a module** (`python -m src....`) so that imports like `from src import project` work.

Everything the demos print is saved in `docs/sample_outputs/*.txt` - compare your console with those files. To regenerate all of them
(about 8-12 minutes) run `python scripts/capture_outputs.py`, then `python scripts/build_results.py` (writes `results.json`).

---------------------------------------------------------------------------------------------------------------------------------------

## 2. Folder map

| Path | Purpose |
|---|---|
| `src/data/make_dataset.py` | seeded generator: income skew + typos, MNAR missing income, correlated pairs, leakage columns, sensitive columns, repeat applicants, production batches |
| `src/data/profile_data.py` | data profile (dtypes, missing, imbalance, skew, correlations, leakage smell test); `--plots` adds two figures (OPTIONAL) |
| `src/features/definitions.py` | one place for column lists, feature kinds, formulas + worked examples |
| `src/features/build_features.py` | **the** feature function `engineer()` (training, batch and API all call it) + the time-based, applicant-disjoint split |
| `src/features/manual_examples.py` | hand calculation vs code for every engineered feature (EMI by hand) |
| `src/features/select_features.py` | filter (variance, correlation, chi-square, ANOVA, mutual information), wrapper (RFE), embedded (Lasso, ElasticNet), governance decision per feature |
| `src/features/registry.py` | reads `configs/feature_registry.yaml`; `check_features_approved()` refuses unapproved features |
| `src/features/leakage_demo.py` | wrong-vs-right demos: leaky columns, scaler fit on all data, target encoding, duplicate applicants |
| `src/models/pipeline.py` | `ColumnTransformer` + `StandardScaler` + model in ONE sklearn Pipeline |
| `src/models/train.py` | baseline -> scorecard candidate -> boosting benchmark, MLflow logging, champion/challenger artifacts |
| `src/models/evaluate.py`, `metrics.py` | ROC-AUC, Gini, KS, PR-AUC, recall at fixed approval rate, calibration, risk bands, money view |
| `src/models/explain.py`, `reason_codes.py` | standardized coefficients, permutation importance, SHAP (OPTIONAL), per-applicant reason codes |
| `src/models/retrain.py` | production batch -> quality -> drift -> decision ladder -> challenger -> same hold-out -> promotion rules -> human approval -> promote / rollback |
| `src/governance/fairness_audit.py` | approval / default rate by gender and age band, adverse impact ratio, equal-opportunity gap, before/after removing proxies |
| `src/governance/age_experiment.py` | "with age vs without age": accuracy and fairness impact + the POLICY decision record |
| `src/governance/model_card.py` | generates `reports/model_review_pack.md` (11 sections incl. sign-off checklist) |
| `src/governance/review_gate.py` | GO / NO-GO gate (6 hard rules); `--demo-bad-model` shows it rejecting a leaky model with AUC 0.95 |
| `src/monitoring/` | `run_drift_check.py` (KS + evidence + ladder), `data_quality.py`, `perf_monitor.py`, `drift_exporter.py` (Prometheus), `simulate.py`, `failure_scenarios.py` |
| `src/api/app.py`, `src/project.py` | shared FastAPI app + THIS project's contract (request/response models, bands, reason codes) |
| `src/common/` | drift (KS), decision ladder, promotion (champion/challenger + registry + rollback), MLflow helpers |
| `configs/` | `feature_registry.yaml` (feature metadata), `selected_features.yaml`, `params.yaml`, `quality_gate.json`, sample requests |
| `docs/DATASET.md`, `docs/REGULATORY_REVIEW_FLOW.md` | data dictionary; development -> validation -> compliance -> committee -> production -> monitoring |
| `docs/sample_outputs/` | REAL captured output of every demo command |
| `notebooks/01_walkthrough.ipynb` | short teaching notebook (all cells executed, outputs stored) |
| `tests/` | data, features, leakage, model, governance, API, monitoring, common (promotion / rollback) tests |
| `dvc.yaml`, `MLproject` | reproducible pipeline (data -> features -> select -> train -> evaluate); MLflow project entry points |
| `Dockerfile`, `k8s/`, `docker-compose.yml`, `monitoring/` | image, Minikube manifests, Prometheus + Grafana stack with alert rules |
| `.github/workflows/ci.yml` | CI: tests -> quality gate -> governance gate -> Docker build |
| `results.json` | machine-readable facts of one full run (numbers, tables, outputs list) |

---------------------------------------------------------------------------------------------------------------------------------------

## 3. The pipeline step by step

Think of each step as a question the project must answer.

| # | Step | Command | Output to read | Question |
|---|---|---|---|---|
| 0 | Generate data | `python -m src.data.make_dataset` | `00_make_data_and_split.txt` | Is the data realistic enough to hurt? |
| 1 | Profile | `python -m src.data.profile_data` | `01_data_profile.txt` | Missing %, skew, imbalance, correlated pairs, leakage smell? |
| 2 | Features by hand | `python -m src.features.manual_examples` | `02_feature_manual_examples.txt` | Can I reproduce every feature with a calculator? |
| 3 | Split + selection | `python -m src.features.build_features` then `python -m src.features.select_features` | `03_feature_selection.txt` | Which features survive statistics AND governance? |
| 4 | Leakage demo | `python -m src.features.leakage_demo` | `04_leakage_demo.txt` | What does a leak look like, and what does it do in production? |
| 5 | Train + MLflow | `python -m src.models.train --bootstrap-champion` | `05_train_mlflow.txt` | Does feature engineering help? Is boosting worth losing explainability? |
| 6 | Evaluate | `python -m src.models.evaluate` | `06_evaluation.txt` | AUC, Gini, KS, calibration, money? |
| 7 | Explain | `python -m src.models.explain` | `07_explainability.txt` | Which features matter? What are the reasons for ONE applicant? |
| 8 | Fairness + age | `python -m src.governance.fairness_audit` and `age_experiment` | `13_fairness_audit.txt`, `14_age_experiment.txt` | Are outcomes comparable across groups? Did removing proxies help? |
| 9 | Review pack + gate | `python -m src.governance.model_card` then `review_gate` | `15_review_pack_and_gate.txt`, `reports/model_review_pack.md` | GO or NO-GO for human review? |
| 10 | Tests + quality gate | `pytest -q` and `python scripts/quality_gate.py` | `08_pytest.txt` | Do the guarantees hold? |
| 11 | Serve + call | `uvicorn src.api.app:app` + `scripts/api_session.py` | `09_api_session.txt` | Does the API give the same answer as training, with reasons? |
| 12 | Monitor | `python -m src.monitoring.run_drift_check --prod data/production/batch_drifted.csv` | `10_`, `11_drift_*.txt` | Did the world change? Does it matter? |
| 13 | Failure scenarios | `python -m src.monitoring.failure_scenarios` | `16_failure_scenarios.txt` | What do the three classic production incidents look like? |
| 14 | Retrain | `python -m src.models.retrain --labels-available [--approve-as "NAME (ROLE)"]` | `12_retrain_promotion.txt` | Is a new model really better, and did a human sign? |

### Key design decisions (read these before the code)

**Prediction-time rule.** Every feature must exist when the application is submitted. Six raw columns do not (`interest_rate_offered`,
`loan_approved`, `days_past_due_after_disbursal`, `collections_status`, `post_disbursal_payment_ratio`, `credit_officer_override`); the registry marks
them Rejected and `check_features_approved()` stops training if one sneaks in. The API request model uses `extra="forbid"`, so a leaky
or sensitive field is rejected with HTTP 422.

**Features** (`src/features/definitions.py`, hand calculation in `02_feature_manual_examples.txt`)

| Feature | Formula | Why a lender cares |
|---|---|---|
| `log_income` | `log(1 + annual_income)` | income is right-skewed (skew 20.8 -> 0.2 after log) |
| `loan_to_income` | `loan_amount / annual_income` | size of the debt relative to earning power |
| `emi_estimate` | `P r (1+r)^n / ((1+r)^n - 1)`, r = 14% / 12 (ASSUMED rate) | what the new loan costs every month; EMI by hand = 14,403.86 for 300,000 over 24 months |
| `emi_to_income`, `debt_service_ratio` | `emi / monthly_income`, `(monthly_debt + emi) / monthly_income` | share of income consumed by repayments |
| `income_missing`, `bureau_score_missing` | 1 if missing | "did not declare income" can itself be informative (MNAR) |
| `employment_stability`, `credit_history_bucket` | ordered buckets | stable, explainable bands |
| `city_tier` | coarse grouping of the 15 cities | the governance-friendly replacement for fine `location` |

Missing values are imputed (median, learned on TRAIN only) **inside the sklearn Pipeline**, so the API handles a missing `annual_income` exactly like training
(a test proves it).

**Selection** (`03_feature_selection.txt`): a correlation filter (|Spearman| > 0.8) removes duplicates (`annual_income` vs `monthly_income`, `monthly_debt` vs
`debt_to_income_ratio`); filter statistics, RFE (wrapper) and L1 / ElasticNet (embedded) vote; a sign check removes features whose coefficient contradicts business
sense (multicollinearity). The final table gives one decision per feature: SELECTED / REJECTED (correlated | weak signal | counter-intuitive sign | leakage |
not available at prediction time | sensitive, policy) / REQUIRES GOVERNANCE (age, location - default: excluded).

**Governance decisions** (policy, not technology): `gender`, `marital_status` never enter the model (gender is used only to audit outcomes); `age` is excluded by
default after the "with age vs without age" experiment (AUC gain +0.0015, not distinguishable from zero, but it lowers approval of the youngest band);
fine `location` is excluded because it behaves as a proxy for gender in this data (see `13_fairness_audit.txt`). Compliance owns these decisions.

**Model choice.** The shipped model is a **logistic-regression scorecard** on 9 selected features.
* *Why not boosting?* `HistGradientBoosting` did not beat it (test AUC 0.7596 vs 0.7683; the required validation margin is 0.02). Even if it had, regulators and applicants
  need reasons: a linear model gives exact, additive reason codes and a stable, auditable score. Boosting stays as a **challenger benchmark** (and SHAP is OPTIONAL).
* *Metrics*: ROC-AUC, **Gini** (= 2 AUC - 1), **KS statistic** (maximum gap between the score distributions of defaulters and good customers - the same two-sample KS idea used for drift),
  PR-AUC (11% positives), recall at a fixed 80% approval rate, a calibration table (is a predicted 10% really 10%?), and a money view with stated assumptions
  (LGD 60%, margin 8%).

**Reason codes.** For one applicant: contribution of each feature = coefficient x standardized value; the top-3 positive contributions (pushing towards default) are
returned by the API as `reason_codes` in plain language. They are what an applicant is told when declined.

**Retraining is never automatic.** Drift is a *signal*. The decision ladder (`src/common/retrain_decision.py`) needs data quality, labelled performance and
business impact. Labels arrive 12 months after disbursal, so a fresh drifted batch gives **INVESTIGATE**, not retrain. Only when labelled performance has fallen
(here AUC -0.069) does the ladder say **RETRAIN_CANDIDATE** - which merely allows training a challenger. The challenger is compared to the champion on the *same, newer* hold-out,
must pass guardrails (KS, recall, fairness AIR >= 0.8, no collapse on the old test set), and a **named human** must sign (`--approve-as`).

---------------------------------------------------------------------------------------------------------------------------------------

## 4. Experiment tracking with MLflow

`train.py` logs three runs (baseline, scorecard candidate, boosting benchmark): parameters, validation/test metrics, the feature registry + selected features
as artifacts (lineage), and the fitted pipeline. The tracking store is a local SQLite file (`mlflow.db`).
```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # open http://localhost:5000
```
On Windows PowerShell use the same command. Compare the runs side by side (select 3 runs -> Compare). Entry points are also available via `MLproject`:
`mlflow run . -e train --env-manager local`.

---------------------------------------------------------------------------------------------------------------------------------------

## 5. DVC - reproducible data + pipeline

```
dvc init                                  # (inside a git repo; add --no-scm to try it without git)
dvc config core.analytics false
dvc repro                                 # runs make_data -> features -> select -> train -> evaluate; unchanged stages are skipped
dvc dag                                   # draws the pipeline
dvc metrics show                          # reports/metrics.json
dvc params diff                           # what changed in configs/params.yaml
# (data/raw is already tracked by the make_data stage, so `dvc add` is NOT needed for it. Use `dvc add <file>` only for a file that is not a stage output.)
mkdir ..\dvc-remote                       # Linux: mkdir ../dvc-remote   (a "remote" can be any folder, a drive or a bucket)
dvc remote add -d localremote ../dvc-remote
dvc push                                  # copy the data versions to the remote; `dvc pull` on another machine gets them back
```
Change `split.train_frac` or `features.assumed_annual_rate_pct` in `configs/params.yaml` and run `dvc repro` again: only the stages that depend on it re-run.
The model review pack records the DVC hash of the data when `dvc.lock` exists (placeholder otherwise).

---------------------------------------------------------------------------------------------------------------------------------------

## 6. Docker

The image needs the trained model, so train first.
```
python -m src.data.make_dataset && python -m src.features.build_features && python -m src.models.train --bootstrap-champion
docker build -t credit-scoring:1.0 .
docker run --rm -p 8000:8000 credit-scoring:1.0
python scripts/smoke_test.py --url http://localhost:8000
```
The image uses `requirements-serving.txt` (no MLflow / SHAP / DVC), runs as a non-root user and has a `HEALTHCHECK` on `/health`.
`python scripts/static_checks.py` verifies the Docker files without Docker (COPY sources exist, the CMD module imports with only the serving packages, k8s labels match).

## 7. Kubernetes with Minikube
```
minikube start
minikube image load credit-scoring:1.0           # the cluster cannot see your local Docker images otherwise
kubectl apply -f k8s/
kubectl get pods                                  # 2 replicas, READY 1/1 after the readiness probe passes
minikube service credit-scoring --url             # prints the URL; then:  python scripts/smoke_test.py --url <that url>
kubectl logs deployment/credit-scoring
# after promoting a new model: rebuild the image (new tag), `minikube image load`, then
kubectl set image deployment/credit-scoring api=credit-scoring:1.1
kubectl rollout undo deployment/credit-scoring    # Kubernetes-level rollback
```

## 8. Observability stack (Prometheus + Grafana)
```
docker compose up --build
```
| Service | URL | What |
|---|---|---|
| API | http://localhost:8000/docs | predictions, `/metrics`, `/health`, `/model-info` |
| Drift exporter | http://localhost:9100/metrics | KS statistic and flag per drift feature (last 500 logged predictions vs reference) |
| Prometheus | http://localhost:9090 | scrapes both; alert rules in `monitoring/alert_rules.yml` (422 rejection spike, p95 latency, high-risk share jump, input drift) |
| Grafana | http://localhost:3000 (admin / admin) | dashboard "ML Service Overview" |

Generate traffic: `python scripts/send_traffic.py --csv data/production/batch_drifted.csv --n 400`, then watch the drift gauges move.
No Docker? `python -m src.monitoring.drift_exporter --once --window 300` prints the same table once from `logs/predictions.jsonl`.

---------------------------------------------------------------------------------------------------------------------------------------

## 9. Drift and retraining runbook

1. **Alert or weekly check**: `python -m src.monitoring.run_drift_check --prod data/production/batch_drifted.csv`
   - shows reference vs production medians, **KS statistic D, p-value, H0, alpha, decision per feature**, plus the extra evidence: data quality, labelled performance, business impact (decision mix).
2. **Read the ladder**:

| Ladder decision | Meaning | Next action |
|---|---|---|
| `NO_ACTION` / `MONITOR` | no meaningful change, or significant-but-small (e.g. `credit_history_years`, D = 0.049, p = 0.0015) | note it; do nothing |
| `FIX_DATA_PIPELINE` | data quality failed (e.g. income unit changed) | fix upstream; do NOT retrain on broken data |
| `INVESTIGATE` | large drift but no labels yet (12-month outcomes) | find the cause (season? upstream? real downturn?), estimate business impact, wait for labels |
| `RETRAIN_CANDIDATE` | drift AND labelled AUC fell >= 0.03 | train a challenger (it must still win) |

3. **Labelled check** (later): `python -m src.monitoring.perf_monitor --prod data/production/batch_drifted.csv` (AUC 0.6998 vs 0.7683, predicted PD 0.159 vs observed 0.217).
4. **Retrain**: `python -m src.models.retrain --labels-available` trains the challenger and prints the comparison and the approval record, then stops at **AWAITING HUMAN APPROVAL**.
5. **Approve**: `python -m src.models.retrain --labels-available --approve-as "A. Rao (Model owner)"` promotes only if every rule passed.
6. **Roll back** if something is wrong: `python -m src.models.retrain --rollback` (restores the model that the current champion replaced; verified over two promote/rollback cycles by a test).
7. Restart the API (or roll the Kubernetes deployment) so that it loads `models/champion.joblib`.

Three classic incidents are reproduced by `python -m src.monitoring.failure_scenarios`: (a) approval rate jumps because an upstream income unit changed -> data-quality failure,
(b) AUC collapses after a "new feature" that is missing in production (leakage), (c) HTTP 422 spike after a schema change.

---------------------------------------------------------------------------------------------------------------------------------------

## 10. CI/CD (GitHub Actions)

`.github/workflows/ci.yml`: on every push / pull request -> install, generate data, train -> `pytest` -> model quality gate (`configs/quality_gate.json`: AUC >= 0.72, KS >= 0.35, recall >= 0.40)
-> **governance gate** (fairness audit -> review pack -> `review_gate`, which fails the build on an unapproved or sensitive feature, AIR < 0.8, missing model card, low AUC, no reason codes)
-> Docker build. Pushing to Docker Hub happens only on `main` and only if the secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` exist. Deployment to a cluster is deliberately manual.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | run from the project root with `python -m ...` (not `python src/...`) |
| `FeatureNotApproved` during training | a feature is not Approved in `configs/feature_registry.yaml` - that is the guard working; do not edit the registry to "make it pass" |
| API returns `503 Model not loaded` | run `python -m src.models.train --bootstrap-champion` first |
| API returns 422 | the body violates the contract (range, unknown category, or a forbidden field such as `gender`); the response tells which field |
| MLflow warns about pickle / `skops` | harmless; models are saved with cloudpickle on purpose (see comment in `train.py`) |
| `dvc repro` asks for git | use `dvc init --no-scm` or `git init` first |
| Windows: `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `docker compose` drift gauges stay empty | send traffic first; the exporter reads `logs/predictions.jsonl` written by the API |
| Training or the leakage demo feels slow | the leakage demo fits ~15 models (1-4 minutes depending on the machine); the full `capture_outputs.py` run takes 8-12 minutes; everything else takes seconds |
| Different numbers than `docs/sample_outputs` | different library versions can change the third decimal; the story (honest ~0.77, leaky ~0.93+, drift flags) must stay the same |

## 12. Core vs optional

| CORE (needed to understand and run the project) | OPTIONAL (extension, never required) |
|---|---|
| generator, features, selection, registry guard, scorecard Pipeline, MLflow, metrics, reason codes, API, drift + ladder, retrain + rollback, pandas-only fairness audit, review pack + gate, tests, Dockerfile, CI, DVC | SHAP on the boosting benchmark, boosting challenger, formal fairness libraries (not used here), Kubernetes/Minikube, Prometheus + Grafana compose stack, `profile_data --plots` |

## 13. What was verified, and where

See the section below.

---------------------------------------------------------------------------------------------------------------------------------------

## Verified in the authoring environment (Linux, Python 3.11, 2 shared CPU cores, no Docker daemon, no Minikube, no internet for DVC remotes)

Everything below was run from a **clean copy** of this repository (data, models, logs, mlruns, reports deleted first):

| Step | Result |
|---|---|
| generate data, split, profile, hand-calculated features, feature selection | PASS (outputs in `docs/sample_outputs/00-03`) |
| train with `--bootstrap-champion` (baseline, scorecard, boosting; MLflow SQLite) | PASS (scorecard test AUC 0.7683, KS 0.4394) |
| `pytest` | PASS - 97 passed (about 17-25 s) |
| `scripts/quality_gate.py`, `src.governance.review_gate` (GO) and `--demo-bad-model` (NO-GO, as intended) | PASS |
| uvicorn on a free port + `smoke_test.py` + `send_traffic.py` (normal and drifted) + `/metrics` | PASS (150 + 150 requests, 0 errors) |
| `drift_exporter` update + served over HTTP | PASS |
| `run_drift_check` normal (NO_ACTION) vs drifted (INVESTIGATE; RETRAIN_CANDIDATE with labels) | PASS |
| `retrain` end to end: INVESTIGATE -> awaiting approval -> promote v2 -> rollback to v1 (+ two-cycle rollback regression test) | PASS |
| notebook: all 12 code cells executed top to bottom (outputs stored) | PASS |
| `dvc init --no-scm && dvc repro` (then a second `dvc repro`: "up to date"; also with git present) | PASS (about 40 s) |
| yaml / json / ipynb validation (19 files) and static Docker / Kubernetes / compose checks (`scripts/static_checks.py`) | PASS |

**NOT run here** (no Docker daemon, no cluster, no GitHub): `docker build` / `docker run`, `docker compose up`, Minikube / `kubectl apply`, the GitHub Actions workflow itself, the Grafana dashboard
rendering, and the PowerShell script `scripts/run_local.ps1` (the Linux equivalent was run command by command). Those files were checked statically only (YAML parses, every `COPY` source exists,
the image's CMD module imports with only the serving packages, Service selector = pod labels, probes use `/health`, Prometheus targets = compose service names).
Latency numbers in `results.json` (mean 70 ms for one request incl. reason codes) were measured on busy shared CPUs; expect lower on a quiet laptop.
