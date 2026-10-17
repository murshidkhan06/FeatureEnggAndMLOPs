# Credit Risk Early Warning (PD) - a banking ML system you can run on your laptop (Project 2)

A bank wants a **monthly early-warning list for existing retail customers** (current account + personal loan). On every **month-end** (`as_of_date`, the *prediction timestamp*) the model
estimates the **probability of default (90+ days past due) within the next 6 months**, so that collections can call the riskiest 10 % *before* the missed payments pile up.

This repo is a small but real-looking ML system: synthetic bank data -> **leakage-safe feature creation from transactions** -> **feature registry with a lifecycle gate** ->
time-based split -> logistic baseline vs gradient boosting -> MLflow -> **monthly batch scoring + REST API** -> Docker / Kubernetes -> monitoring (Prometheus, Grafana, KS drift)
-> **drift handling with label lag** -> **retraining with human approval and rollback**.

```
DATA -> FEATURE ENGINEERING -> TRAINING -> EXPERIMENT TRACKING -> REGISTRY (champion/challenger) -> DEPLOYMENT (batch + API)
     -> MONITORING -> DRIFT DETECTION -> RETRAINING -> DEPLOYMENT ...
```

**What this project teaches that the others do not:** *time as a correctness problem*. A feature belongs to a timestamp; a feature that quietly looks past that timestamp
makes the model look brilliant in the lab and fail in production. You will reproduce that failure, then fix it with an availability rule, a leakage-safe pipeline,
a **future-perturbation test** and a governance gate.

> All data is synthetic (seeded, `docs/DATASET.md`). Numbers are plausible for teaching, not facts about any bank. ~34 % of the simulated customers default over 54 months
> (deliberately high so each monthly snapshot has enough defaulters; ~4 % of snapshot rows are positives).

---------------------------------------------------------------------------------------------------------------------------------------

## 1. Quick start (about 3 minutes)

**Windows PowerShell** (Python 3.11 recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data.make_dataset                   # 1. raw tables: customers, transactions, loan_status, defaults   (~5 s)
python -m src.features.build_features             # 2. leakage-safe monthly snapshots + production batches          (~10 s)
python -m src.features.select_features            # 3. choose features on TRAINING rows only                        (~6 s)
python -m src.models.train --bootstrap-champion   # 4. baseline + candidate, MLflow, install champion v1             (~12 s)
pytest -q                                         # 5. 108 tests                                                    (~30 s)
python scripts/quality_gate.py                    # 6. CI-style minimum bar
uvicorn src.api.app:app --port 8000               # 7. serve (Swagger UI: http://localhost:8000/docs)
```
(or simply `.\scripts\run_local.ps1`)

**Linux / macOS / WSL** - the same commands; activate with `source .venv/bin/activate` (or `bash scripts/run_local.sh`).

In a second terminal:
```
python scripts/smoke_test.py --url http://localhost:8000                                   # health + one valid + one invalid request
python scripts/send_traffic.py --csv data/production/batch_normal.csv --n 200 --shuffle
python -m src.models.batch_score                                                            # the realistic mode: score ALL customers -> reports/monthly_scores.csv
```
Always run python **from the project root as a module** (`python -m src....`) so that imports like `from src import project` work.

---------------------------------------------------------------------------------------------------------------------------------------

## 2. Folder map

| Path | Purpose |
|---|---|
| `src/data/make_dataset.py` | seeded generator of the 4 raw tables with injected problems (missing values, outliers, late-arriving data, leakage columns, proxy attribute, macro stress) |
| `src/data/profile_data.py` | data profile: shapes, imbalance, missing values, correlations, outliers, segment proxy, label lag |
| `src/features/definitions.py` | feature names, windows, **availability lags** (one place for training, batch and API) |
| `src/features/build_features.py` | `build_snapshot_features(as_of)` (filters by `available_at <= as_of` FIRST), `assert_no_future_data()`, derived features, target, time split |
| `src/features/leakage_checks.py` | the **future-perturbation test** |
| `src/features/leakage_demo.py` | the leakage lesson in four acts (leaky balance, post-default columns, the fix, random vs time split) |
| `src/features/manual_examples.py` | hand calculation vs code on one tiny customer |
| `src/features/select_features.py` | filter / wrapper / embedded selection on training rows + out-of-time veto |
| `src/governance/feature_lifecycle.py` + `configs/feature_registry.yaml` | feature metadata, lifecycle history, **Production gate**, `check_features_approved()` + CLI (`table`, `history`, `validate`, `gate-demo`) |
| `src/models/pipelines.py`, `train.py` | sklearn Pipelines (imputer + indicator + scaler + model), baseline vs candidate, MLflow, champion rule |
| `src/models/evaluate.py`, `explain.py` | ROC-AUC, PR-AUC, recall@top-10 %, lift, KS + business meaning; permutation importance (SHAP optional) |
| `src/models/batch_score.py` | **monthly batch scoring** of every customer -> `reports/monthly_scores.csv` |
| `src/models/retrain.py` | **the retraining workflow** (quality -> drift -> ladder -> challenger -> same hold-out -> rules -> approval record -> promote/rollback) |
| `src/common/` | artifact I/O, KS drift, decision ladder, promotion + registry, MLflow helpers, paths, params, facts |
| `src/api/app.py`, `src/project.py` | FastAPI app (shared) + the project contract (request/response ranges, risk bands, derived features) |
| `src/monitoring/` | `run_drift_check` (KS + ladder), `data_quality`, `proxy_metrics` + `perf_monitor` (label lag), `credit_policy` (bank rules), `drift_exporter` (Prometheus) |
| `configs/` | `feature_registry.yaml`, `selected_features.yaml` (generated), `quality_gate.json`, `sample_request.json` (tunable numbers are in `params.yaml` at the project root, DVC's default) |
| `tests/` | 108 tests: data, features, leakage, governance, model, API, batch scoring, monitoring, retraining, configs |
| `notebooks/01_walkthrough.ipynb` | short teaching notebook; every code cell is executed by `scripts/run_notebook_cells.py` |
| `docs/DATASET.md`, `docs/sample_outputs/` | data dictionary; the **real console output** of every demo command |
| `dvc.yaml`, `MLproject`, `Dockerfile`, `docker-compose.yml`, `k8s/`, `monitoring/`, `.github/workflows/ci.yml` | pipeline, packaging, deployment, observability, CI |
| `scripts/` | `quality_gate`, `smoke_test`, `send_traffic`, `api_session`, `traffic_demo`, `static_checks`, `capture_outputs`, `build_results`, `run_notebook_cells`, `run_local.*` |
| `results.json` | machine-readable facts (numbers, tables, lists) for the course guide |
| `data/`, `models/`, `logs/`, `reports/`, `mlflow.db`, `mlruns/` | **generated** (git-ignored) |

---------------------------------------------------------------------------------------------------------------------------------------

## 3. The one rule: "would this be known on the day we score the customer?"

* **Prediction timestamp** = `as_of_date` (a month-end). Everything the model sees must be knowable on that date.
* **Availability rule** = a row may be used only if `available_at <= as_of_date`. Transactions are available the day they happen (lag 0). `loan_status` for month M is **published on day 5 of M+1**,
  so on 2023-02-28 the February row (available 2023-03-05) must NOT be used; on 2023-03-31 it may.
* **Leakage-safe pipeline** = `build_snapshot_features(as_of)` **filters the raw tables first**, then computes features; `assert_no_future_data()` is a seat-belt on every table.
* **Future-perturbation test** = corrupt every row that is not yet available (211,924 future transactions + unpublished loan_status at 2022-06-30), rebuild the features: a safe pipeline returns
  *identical* numbers. `tests/test_leakage.py` runs it for the safe pipeline (must pass) and for the 4 leaky features (must fail).

| Feature (all use only rows with `available_at <= as_of`) | Window | Formula | Hand example (checked in `02_feature_manual_examples.txt`) |
|---|---|---|---|
| `average_balance` | 90 days | mean `balance_after` | 10 transactions with balances 80,80,180,160,75,75,175,155,70,70 (thousand) -> 112,000 |
| `balance_trend` | 6 months | OLS slope of 6 month-end balances | `[100, 80, 60]` -> ((-1)(20)+0+(1)(-20)) / 2 = **-20** |
| `salary_credit_frequency` | 6 months | salary credits / 6 | 5 credits -> 0.8333 |
| `loan_payment_ratio` | 6 months | payments made / due | 5 / 6 -> 0.8333 |
| `missed_payment_count` | 6 published months | sum of `missed_payment` | one missed instalment; the unpublished March row is ignored (naive sum would say 2) |
| `credit_utilization` | 6 months | mean monthly card spend / limit | 6,000 / 30,000 -> 0.2 |
| `recent_transaction_count` | 30 days | number of transactions | 4 |
| `income_stability` | 6 months | std / mean of salary amounts (coefficient of variation) | salaries 100,110,90,100,100 -> 7.07 / 100 = **0.0707** |
| `account_age` | static | months since signup | 36 |
| `util_x_missed`, `balance_trend_norm`, `log_average_balance` | derived | utilisation x missed; trend / (abs balance + 1); log1p(balance) | `add_derived_features()` - the same function offline and in the API |

---------------------------------------------------------------------------------------------------------------------------------------

## 4. The pipeline, step by step

Every command prints a lot on purpose (it is a teaching tool). The captured output of each is in `docs/sample_outputs/`.

| # | Command | What you learn | Captured output |
|---|---|---|---|
| 0 | `python -m src.data.make_dataset` then `python -m src.features.build_features` | raw tables (324k transactions), 49,764 monthly snapshots, time split, production batches | `00_make_dataset.txt` |
| 1 | `python -m src.data.profile_data` | imbalance (3.8 %), missing values (card limit = MAR, salary = MNAR), correlated features (|r| 0.84), outliers, segment proxy | `01_data_profile.txt` |
| 2 | `python -m src.features.manual_examples` | hand calculation = code for all 12 features on one toy customer | `02_feature_manual_examples.txt` |
| 3 | `python -m src.features.select_features` | filter vs wrapper vs embedded votes; out-of-time veto drops `account_age` | `03_feature_selection.txt` |
| 4 | `python -m src.features.leakage_demo` | **the leakage lesson**: wrong vs right with numbers | `04_leakage_demo.txt` |
| 5 | `python -m src.models.train --bootstrap-champion` | baseline vs candidate, MLflow, champion rule | `05_train_mlflow.txt` |
| 6 | `python -m src.models.evaluate` | metrics + business meaning + decile table | `06_evaluation.txt` |
| 7 | `python -m src.models.explain [--shap]` | standardised coefficients, permutation importance, SHAP (OPTIONAL) | `07_...txt`, `07b_..._OPTIONAL.txt` |
| 8 | `pytest -v` | 108 tests | `08_pytest.txt` |
| 9 | `python scripts/api_session.py` | scripted API session + latency | `09_api_session.txt` |
| 10-11 | `python -m src.monitoring.run_drift_check --prod data/production/batch_{normal,drifted,dq_issue}.csv --today <date>` | KS drift + ladder; the label-lag problem; data-quality failure | `10_...`, `11_...`, `11b_...`, `11c_label_lag_timeline.txt` |
| 12 | `python -m src.models.retrain ...` | retraining workflow incl. approval + rollback | `12_retrain_promotion.txt` |
| 13 | see section 8 | the three failure scenarios | `13_failure_scenarios.txt` |
| 14 | `python -m src.models.batch_score` | monthly batch scoring (986 customers on 2025-06-30, no labels exist yet) | `14_batch_score.txt` |
| 15 | `python -m src.governance.feature_lifecycle table|history|validate|gate-demo` | the registry and the Production gate | `15_feature_registry.txt` |
| 16-18 | `python scripts/quality_gate.py`, `python scripts/traffic_demo.py`, `python scripts/static_checks.py` | gate, drift exporter on live traffic, static Docker/K8s checks | `16_...`, `17_...`, `18_...` |

### 4.1 Headline results (measured; see the output files for the full tables)

**Time-based split** (train 2021-06..2022-03, purge 2022-04..08, validation 2022-09..11, test 2022-12..2023-02; test touched once)

| model | period | ROC-AUC | PR-AUC | recall@top-10 % | lift@10 % | KS of score |
|---|---|---|---|---|---|---|
| baseline: logistic regression | validation | 0.890 | 0.544 | 0.660 | 6.6 | 0.647 |
| | test | 0.882 | 0.601 | 0.674 | 6.7 | 0.636 |
| **candidate: HistGradientBoosting (champion v1)** | validation | 0.897 | 0.529 | 0.631 | 6.3 | 0.679 |
| | **test** | **0.905** | **0.618** | **0.694** | **6.9** | **0.658** |

**Business meaning (test period):** of 3,554 customers scored, collections calls the top 10 % = 356 people. 147 customers really defaulted within 6 months; the list reaches **102 (69 %)** and misses 45.
Finding one defaulter costs **3.5 calls** with the model versus **24.2 calls** by random calling.

**Leakage numbers** (`04_leakage_demo.txt`)

| | train | validation | test | **live** (2023-03..08, labels 6 months later) |
|---|---|---|---|---|
| RIGHT: legitimate features only (HistGradientBoosting) | 0.937 | 0.886 | 0.885 | **0.834** |
| WRONG: + `average_balance_leaky` | 0.980 | 0.942 | **0.956** | **0.775** |

The single leaky column lifts test AUC by 0.07 (single-feature AUC 0.913 versus 0.645 for the honest `average_balance`), and in production the "great" model is **worse than the honest one** (0.775 vs 0.834).
Second, subtler leak: `days_past_due` of the target period gives AUC **1.00** (default is *defined* as 90+ dpd), `collections_status_latest` 0.96 and `recoveries_target_period` 0.94 when added to a model.
A **random** split of the snapshots reports 0.911 versus 0.885 for the honest time split (the same customer sits on both sides).

### 4.2 Model choice (why not something fancier?)
* **Baseline first.** Logistic regression (median imputation + missing indicator + scaler, L2) is the scorecard workhorse of credit risk: coefficients are explainable to a regulator.
* **Candidate: HistGradientBoosting.** Credit risk has *threshold* effects (habitual card use above ~35 % utilisation, a thin cash buffer) and interactions that a straight line cannot learn; boosting learns them,
  handles missing values natively and trains in seconds. The rule in `train.py`: **keep the simple model unless the candidate beats it on validation by >= 0.005 ROC-AUC** (it did: +0.0072).
* **Not** a deep network, LSTM on raw transactions or a hyper-parameter search (CORE ends at a sensible default; a small search is an exercise): ~12k training rows with 450 defaulters do not justify the complexity,
  and every extra moving part is something model-risk has to validate.
* **Metrics.** Accuracy is useless at 4 % positives. We report ROC-AUC (ranking), PR-AUC (honest with rare positives), **recall@top-10 %** and **lift** (collections capacity) and **KS of the score**
  (credit-scoring standard). Calibration is checked separately in monitoring (`mean PD / observed rate`).
* **Feature selection** (`03_feature_selection.txt`): a feature is kept if >= 2 of 3 families keep it (filter: variance + ANOVA + mutual information + correlation pruning; wrapper: RFE; embedded: L1 logistic).
  `missed_payment_count` is dropped by the correlation filter (|r| = 0.84 with `loan_payment_ratio`) but kept by the other two. `account_age` is kept by all three families and then **vetoed out-of-time**
  (shuffling it *improves* validation AUC: it acts as a customer fingerprint). `average_balance` is replaced by its derived `log_average_balance`. Final: **10 of 12** features.

### 4.3 Explainability (`07_explainability.txt`)
Permutation importance on the test period (ROC-AUC drop when a column is shuffled): `credit_utilization` 0.050, `loan_payment_ratio` 0.034, `income_stability` 0.025, `balance_trend_norm` 0.011.
Correlated features share credit (`missed_payment_count` looks weak because `loan_payment_ratio` tells the same story). Importance describes the **model**; it does not prove that lowering utilisation *causes* lower default risk.

---------------------------------------------------------------------------------------------------------------------------------------

## 5. Feature governance (registry + lifecycle gate)

`configs/feature_registry.yaml` is the paper trail a model-risk team expects. **Every** feature (17 entries incl. 4 leaky ones and the sensitive `segment`) has: name, definition, source, data type, owner,
calculation logic, window, **availability (lag + rule)**, prediction-time availability, status (Approved / Rejected + reason), stage, version and a **lifecycle history** (Development -> Validation -> Approval -> Production with date and approver).

```
python -m src.governance.feature_lifecycle table       # the registry as a table
python -m src.governance.feature_lifecycle history     # who approved what, when
python -m src.governance.feature_lifecycle validate    # re-run the REAL checks on today's data
python -m src.governance.feature_lifecycle gate-demo   # watch the gate block a feature
```
* A feature may enter Approval/Production only if the **Validation record** shows: leakage test pass, null rate <= 20 %, availability rule declared, owner assigned. Illegal jumps (Development -> Production) are refused.
* **`average_balance_leaky` v1 is Rejected** ("includes the post-default balance collapse; fails the future-perturbation test") and **replaced by `average_balance` v2** (stops at `as_of_date`).
* `segment` is Rejected as a sensitive / proxy attribute (needs a fairness review before use).
* **Training refuses non-Production features**: `train.py`, `retrain.py` and `select_features.py` call `check_features_approved()`; try adding `average_balance_leaky` to `configs/selected_features.yaml` and run `train` - it stops with the reason.

---------------------------------------------------------------------------------------------------------------------------------------

## 6. Batch scoring vs real-time API (same model, same derived-feature code)

| | Monthly batch (the realistic mode) | Real-time API |
|---|---|---|
| command | `python -m src.models.batch_score [--as-of 2025-06-30]` | `POST /predict` |
| unit | every eligible customer -> ranked list `reports/monthly_scores.csv` (PD, risk band, rank, `contact_recommended`) | one customer |
| input | raw tables, filtered by the availability rule | the **engineered feature vector** a feature-store lookup would supply + `customer_id` + `features_as_of` |
| when | after month-end (collections capacity = top 10 %) | ad-hoc (a relationship manager opens one customer) |

**API** (`/docs` for Swagger). Request: `customer_id, features_as_of, average_balance, balance_trend, salary_credit_frequency, loan_payment_ratio, missed_payment_count, credit_utilization (nullable), recent_transaction_count,
income_stability (nullable), account_age` with pydantic ranges (`Field(ge=, le=)`: out-of-range = HTTP 422, the data contract). Response: `probability_of_default_6m, risk_band (LOW < 5 % <= MEDIUM < 12 % <= HIGH),
contact_recommended, model_version, features_as_of` (timestamp echo). Extra: `GET /health`, `/model-info`, `/metrics` (Prometheus). `customer_id` is echoed but **never written to the prediction log** (privacy); the log holds model inputs only.
Measured locally over 200 sequential requests (client side, includes HTTP): p50 ~17-20 ms, p95 ~27-32 ms on an idle laptop-class VM; p50 ~50-60 ms, p95 ~140-220 ms in the final verification run, when the 2-core VM was shared with other jobs. The model itself is not the bottleneck; expect your numbers to move with CPU load.

**No training-serving skew:** `request_to_frame()` adds the 3 derived features with the *same* function as training; `tests/test_api.py` asserts online features equal offline features and
`tests/test_batch_score.py` asserts that the monthly batch pipeline reproduces the training-snapshot scores (to 4 decimals).

---------------------------------------------------------------------------------------------------------------------------------------

## 7. MLflow, DVC, Docker, Kubernetes, observability, CI

### MLflow UI
```
mlflow ui --backend-store-uri sqlite:///mlflow.db        # then open http://127.0.0.1:5000
```
Experiment `credit_risk_pd_model`: one run per model (baseline, candidate) with parameters (periods, features, data hash, hyper-parameters), train/val/test metrics, the registry and selected-features artifacts, and the logged sklearn pipeline.

### DVC (pipeline as a graph)
```
dvc init --no-scm            # (or `dvc init` inside a git repo)
dvc config core.analytics false
dvc repro                    # make_data -> features -> select -> train -> evaluate   (~30 s)
dvc dag                      # draw the graph
dvc metrics show             # reports/evaluation.json
dvc params diff              # after editing params.yaml (needs a git repository)
# version a data file and use a LOCAL remote (no cloud account needed):
dvc add reports/monthly_scores.csv       # after `python -m src.models.batch_score`: version a result file
dvc remote add -d localstore ../dvc-store
dvc push
```
`dvc.yaml` declares dependencies, params and outputs of each stage; change a value in `params.yaml` (e.g. `model.hgb.max_iter`) and only `train` and `evaluate` re-run.
A promotion changes `models/champion.joblib`, so `dvc status` will later report `train` as changed - that is the point of a champion that moves.

### Docker
```
python -m src.data.make_dataset && python -m src.features.build_features && python -m src.features.select_features && python -m src.models.train --bootstrap-champion
docker build -t credit-risk-pd:1.0 .
docker run -p 8000:8000 credit-risk-pd:1.0
python scripts/smoke_test.py --url http://localhost:8000
```
The image contains `src/`, `models/champion.joblib`, `data/processed/reference.csv` and `configs/`; it runs as a non-root user with a health check. Run `python scripts/static_checks.py` before building.

### Kubernetes with Minikube
```
minikube start
docker build -t credit-risk-pd:1.0 .
minikube image load credit-risk-pd:1.0          # the cluster cannot see your local Docker images otherwise
kubectl apply -f k8s/
kubectl get pods                                # wait for 2/2 Ready
minikube service credit-risk-pd --url           # prints the URL; then run scripts/smoke_test.py --url <that url>
kubectl delete -f k8s/                          # clean up
```
### Observability stack (Docker Compose)
```
docker compose up --build        # API :8000, Prometheus :9090, Grafana :3000 (admin/admin), drift exporter :9100
python scripts/send_traffic.py --csv data/production/batch_normal.csv --n 300 --shuffle
python scripts/send_traffic.py --csv data/production/batch_drifted.csv --n 300 --shuffle
```
Grafana dashboard "ML Service Overview": request rate, errors, latency, predictions by risk band, score histogram, KS statistic and flag per feature. A flag means "the inputs look different from training" - a **signal**.
(On windows of a few hundred requests a feature with D close to 0.10, such as `account_age`, can flicker; the monthly check with the full batch is the authority.)
`python scripts/traffic_demo.py` does the same on your laptop without Docker and prints the exporter gauges.

### CI/CD (GitHub Actions, `.github/workflows/ci.yml`)
`test` job: install -> generate data -> features -> selection -> train -> `pytest` -> quality gate (`configs/quality_gate.json`) -> leakage demo must still pass.
`build` job: rebuild the model, run the static checks, build the Docker image; the push step runs only on `main` and only if Docker Hub secrets exist. Deployment to a cluster is intentionally manual (needs credentials).

---------------------------------------------------------------------------------------------------------------------------------------

## 8. Drift, label lag and retraining runbook

### 8.1 What happens in the stress period (`11_drift_shifted.txt`, batch of 2024-01-31 vs the training reference)

KS test in one paragraph: **H0** = reference and production values come from the same distribution. `D` (0..1) is the largest gap between the two empirical CDFs (the *effect size*); the **p-value** says how surprising
such a gap would be if H0 were true; alpha = 0.05. With ~1,100 customers even tiny gaps are significant, so we also require `D >= 0.10` ("significant but small" is real but not worth acting on).
A large p-value is **not** proof of no drift.

| feature | ref mean | prod mean | D | p-value | verdict |
|---|---|---|---|---|---|
| income_stability | 0.049 | 0.073 | **0.427** | 2e-119 | drift |
| balance_trend | -257 | -3,749 | **0.295** | 2e-56 | drift |
| salary_credit_frequency | 0.965 | 0.908 | **0.238** | 2e-36 | drift |
| credit_utilization | 0.225 | 0.256 | **0.152** | 1e-12 | drift |
| account_age | 144.9 | 147.5 | 0.084 | 7e-05 | significant but small -> watch only |
| average_balance | 134,401 | 130,188 | 0.030 | 0.51 | no evidence of drift |

**Significant-but-small example (should trigger nothing):** `account_age` on the *normal* batch of 2023-03-31: D = 0.056, p = 0.018. Decision: `MONITOR`.

### 8.2 The label-lag problem: drift is seen immediately, the truth arrives 6 months later
The label `default_next_6m` of a customer scored on 2024-01-31 is known only on **2024-07-31**. Same batch, same drift table, three different "today" dates (`--today`, all in `11_drift_shifted.txt`):

| today | what exists | decision |
|---|---|---|
| 2024-02-29 | drift + score shift only (mean PD x2.06; **27.6 % of customers above the contact threshold vs 10 % planned** = collections overloaded). Labels: none. | **INVESTIGATE** - find the cause, estimate impact, wait for evidence |
| 2024-05-15 | + **pseudo-label**: early-delinquency rate (missed instalment in the 3 months after as_of) 16.8 % vs 8.1 % normal = **x2.08** | **RETRAIN_CANDIDATE** (>= 2 key features drift AND outcome evidence) |
| 2025-06-30 | + **real labels**: ROC-AUC 0.8245 (drop 0.080 from 0.9046), recall@10 % 0.51 vs 0.69, calibration x1.14 | **RETRAIN_CANDIDATE** |

`python -m src.monitoring.perf_monitor --today 2024-02-29` prints the month-by-month timeline of which months can already be graded (`11c_label_lag_timeline.txt`).

### 8.3 The decision ladder (cheapest check first) - `src/common/retrain_decision.py` + `src/monitoring/credit_policy.py`
1. **Data quality** (schema, one snapshot date, duplicates, ranges, null rate vs reference). Failed -> `FIX_DATA_PIPELINE`; never retrain on broken data.
2. **Drift** (KS on the 9 inputs). Count the **key features** (`salary_credit_frequency, credit_utilization, income_stability, missed_payment_count, balance_trend`) with significant AND large drift.
3. **Outcome evidence**: labelled AUC drop >= 0.05 or calibration x1.5 (needs labels), **or** pseudo-label ratio >= 1.5 (needs ~3 months).
4. **Rule:** `RETRAIN_CANDIDATE` if (drift in >= 2 key features AND outcome evidence) **or** a scheduled quarterly retrain is due (`--scheduled`). Otherwise `INVESTIGATE`, `MONITOR` or `NO_ACTION`.
`RETRAIN_CANDIDATE` only means *train a challenger*. **Drift alone never retrains** (`tests/test_monitoring.py`, `tests/test_common.py`).

### 8.4 The retraining workflow (`12_retrain_promotion.txt`)
```
python -m src.models.retrain --prod data/production/batch_drifted.csv                                  # stops at "AWAITING APPROVAL"
python -m src.models.retrain --prod data/production/batch_drifted.csv --approve-as "Priya Nair (Model Risk)"   # promotes if the rules pass
python -m src.models.retrain --rollback                                                                # restore the previous champion
```
Steps: data quality -> KS drift -> ladder (stop unless `RETRAIN_CANDIDATE`) -> train a challenger on the refreshed window (**only snapshots whose 6-month label is already known**, purge gap of 6 months before the hold-out) ->
score champion **and** challenger on the **same newer hold-out** (2024-10..12, neither was trained on it) -> `promotion.compare` (ROC-AUC must improve by >= 0.005 AND recall@10 % may not fall by more than 0.03) ->
print an **approval record** (`reports/approval_record.json`) -> promote **only** if the rules pass **and** a human name is given; the old champion is archived and `--rollback` restores it.

What the captured run shows: champion v1 on the hold-out has ROC-AUC 0.8005 and recall@10 % 0.510 and flags **32 % of customers** (plan: 10 %); challenger v2 (rolling window, 41,067 rows) has 0.8157 / 0.546 and flags 10.3 %.
Rules pass -> without a name: **AWAITING APPROVAL**; with `--approve-as`: **PROMOTED**; with `--min-improvement 0.05` the rules **fail** and even a signed approval cannot promote.
The normal batch stops at `MONITOR`; the data-quality batch stops at `FIX_DATA_PIPELINE` - no challenger is trained for either.

*Note:* a registry `roc_auc` is measured on that model's own evaluation period (v1: calm test months; v2: the harder stress hold-out), so compare models only on the same hold-out, as step 5 of the script does.
After a promotion copy `data/processed/reference_<version>.csv` over `reference.csv` and restart the API (or rebuild the image and `kubectl rollout restart deployment/credit-risk-pd`).

---------------------------------------------------------------------------------------------------------------------------------------

## 9. Failure scenarios reproduced (`13_failure_scenarios.txt`; tests in `tests/`)

| Scenario | Symptom | Detection | Fix |
|---|---|---|---|
| **(a) Leaky feature in training** | test AUC 0.956, live AUC 0.775 (below the honest model, 0.834) | future-perturbation test fails; registry marks the feature Rejected; training refuses it | rebuild features at `as_of_date`; availability rule; keep the gate in CI |
| **(b) Null rate of `salary_credit_frequency` jumps** (source-system migration): 40 % empty vs 0 % | KS shows *no* drift (nulls are dropped); model scores silently degrade | **data-quality gate** (+40 pp > 5 pp) -> `FIX_DATA_PIPELINE`, `retrain` stops | fix the pipeline; do not retrain on broken data |
| **(c) Challenger trained on the stress period only** | 9,189 rows / 657 defaulters instead of 37,513 / 1,913 (calm test months held out of both): AUC -0.004 on the stress hold-out and **-0.018 on calm data** vs the rolling window | `retrain --window stress_only --calm-check` | cover every regime the model will meet; treat "latest months only" as a hypothesis to test |
| Label lag | performance unknown for 6 months | `perf_monitor`, proxy metrics | pseudo-labels, score-shift and contact-rate monitoring |
| Unpublished data | one future `loan_status` row inflates `missed_payment_count` (1 -> 2) | `02_feature_manual_examples.txt` | availability rule |

---------------------------------------------------------------------------------------------------------------------------------------

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | run from the project root with `python -m src....` (not `python src/...py`) |
| `FileNotFoundError: data/processed/snapshots.csv` | run `python -m src.features.build_features` (step 2) |
| `FeatureNotApprovedError` during train | the feature is Rejected / unknown in `configs/feature_registry.yaml` - this is the gate working |
| `FileNotFoundError: models/champion.joblib` | `python -m src.models.train --bootstrap-champion` |
| `/predict` returns 422 | a value is outside the data contract (e.g. `credit_utilization` > 3, `features_as_of` missing); the response says which field |
| `mlflow ui` shows nothing | start it from the project root with `--backend-store-uri sqlite:///mlflow.db` |
| MLflow pickle warning at train time | expected: sklearn models are stored with cloudpickle |
| sklearn `FutureWarning: 'penalty' was deprecated` | harmless on scikit-learn 1.8 (silenced in the selection step); the code works on 1.4+ |
| Windows: `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process Bypass` |
| `minikube image load` is slow | normal for the first load; check `minikube image ls` |
| Docker build fails at `COPY models/champion.joblib` | train first (generated files are git-ignored) |
| `dvc repro` re-runs everything | a dependency changed (e.g. `configs/feature_registry.yaml`); `dvc dag` / `dvc status` show which |
| `dvc params diff` says "Only supported for Git repositories" | you used `dvc init --no-scm`; use a git repo for that command |
| Container logs `InconsistentVersionWarning` | the scikit-learn in the image differs from the one that trained the model; pin the same version in `requirements-serving.txt` |

---------------------------------------------------------------------------------------------------------------------------------------

## 11. CORE vs OPTIONAL

* **CORE**: availability rule + leakage-safe pipeline + future-perturbation test, feature registry with lifecycle gate, time split with purge, logistic baseline -> gradient boosting, the business metrics,
  permutation importance, API + monthly batch scoring, KS drift + data-quality gate + decision ladder with label-lag proxies, champion/challenger retraining with approval and rollback, MLflow, DVC, Docker, Kubernetes, CI.
* **OPTIONAL** (never needed by the core pipeline; marked `OPTIONAL`): SHAP (`python -m src.models.explain --shap`), a gradient-boosting hyper-parameter search (exercise; not implemented), the docker-compose observability stack, Minikube.

## 12. Exercises
1. In `src/features/manual_examples.py` pay the January salary and loan instalment. Predict, then check, the changes in `salary_credit_frequency`, `income_stability`, `loan_payment_ratio`, `missed_payment_count`.
2. Add a new feature `salary_to_instalment` to the registry (all fields!), build it, and push it through Development -> Validation -> Approval -> Production with `feature_lifecycle.advance()`. What stops you if you skip the leakage test?
3. Change the leaky window in `leakage_demo.py` from +7 to +1 month. How does the single-feature AUC change, and why?
4. Run `perf_monitor` for `--today` values 2024-04-30, 2024-07-31, 2024-10-31. When would *you* have retrained, and what did you know at that time?
5. Raise `drift.min_effect` to 0.20 in `params.yaml` and re-run the drift check. Which features change status? Who should own that threshold?
6. (Optional) Add a small grid search over `learning_rate` and `max_depth` for HistGradientBoosting using the validation period only. Did it clear the +0.005 bar again?

## 13. What was verified in the authoring environment (honest note)
Authored and verified on Linux, Python 3.11, pandas 3 / numpy 2 / scikit-learn 1.8 / mlflow 3, from a **clean copy** (data, models, logs, MLflow store, reports deleted first), in this order:
data generation, features + selection, training with `--bootstrap-champion`, `pytest` (**108 passed**), quality gate, a real `uvicorn` server with `scripts/smoke_test.py`, `scripts/send_traffic.py` (normal and drifted), `curl /metrics`,
the drift exporter (one update and served over HTTP), `run_drift_check` (normal vs drifted vs data-quality batch), `retrain.py` end to end including approval and rollback, every notebook code cell executed,
`dvc init --no-scm && dvc repro` (works offline, ~30 s, second run skips all stages), and the static checks (yaml/json parse, every Docker `COPY` source exists, the CMD module and the drift exporter import without development-only packages, Service selector = pod labels, image names agree).
**Not run here** (no Docker daemon, no Minikube, no GitHub): `docker build/run`, `docker compose up`, `minikube`/`kubectl apply`, GitHub Actions. Those files are validated statically only.
Expect to fix small environment-specific issues (proxy, Docker Desktop resources) the first time you run them. Timings are from a small Linux VM. Every number in this README comes from `docs/sample_outputs/` and `results.json` (seeded, so you should reproduce them exactly on the same library versions; small differences are possible with other scikit-learn versions).
