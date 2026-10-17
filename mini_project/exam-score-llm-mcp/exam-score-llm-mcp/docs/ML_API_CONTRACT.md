# The existing ML API — contract this project depends on

Extracted by reading `exam-score-mlops/api/main.py` and the live `/openapi.json` (a saved copy is in
`tests/fixtures/ml_api_openapi.json`; an integration test compares it against the running API, so drift is caught).

* Title: *Student Exam Score Predictor* · version 1.0.0 · port **8000**
* Endpoints: `POST /predict`, `GET /health`, `GET /metrics`, `GET /openapi.json`, `GET /docs`

## `POST /predict`

Request body (`StudentInput`) — **all 10 fields required, no defaults**, raw unprocessed values:

| Field | Type | Constraint | Meaning | Does the model use it? |
|---|---|---|---|---|
| `study_hours` | number | 0 – 40 | Weekly study hours | yes |
| `attendance_pct` | number | 0 – 100 | Attendance % | yes |
| `mock_test_1` | number | 0 – 100 | Mock test score | yes |
| `mock_test_2` | number | 0 – 100 | Mock test score | yes |
| `mock_test_3` | number | 0 – 100 | Mock test score | yes |
| `income_bracket` | string | `Low` / `Medium` / `High` | Family income bracket | yes |
| `city` | string | (training: Delhi, Pune, Mumbai, Bengaluru) | City | yes |
| `enrollment_date` | string | `YYYY-MM-DD` | Enrollment date (the pipeline derives *tenure*) | yes |
| `shoe_size` | number | — | **Deliberately irrelevant** column | **dropped by the pipeline** |
| `lucky_number` | integer | — | **Deliberately irrelevant** column | **dropped by the pipeline** |

Response `200`: `{ "predicted_final_score": 95.2, "model_version": "…", "request_id": "<uuid made by the ML API>" }`
(score rounded to 1 decimal). `422` for invalid input, `503` if the model is not loaded.

Feature engineering (derived `tenure_days`, `study_x_attendance`, imputation, scaling, encoding, PCA, selection) happens
**inside** the pipeline. This project sends only the 10 raw fields — it never re-implements any preprocessing.

## What this means for the chatbot

* The API has no defaults, so the chatbot must obtain **all 10** values from the user. It asks for `shoe_size` and
  `lucky_number` too (explaining the model ignores them) rather than inventing them.
* There is no "assignments" feature. If a user mentions assignments, that information is simply not an input; if the LLM tries to pass an `assignments` field anyway, the orchestrator rejects the unknown field.
* `study_hours` is per **week**; the data's study-hours range was ≈ 0.7–21.7, so very large values are extrapolation.
* Final scores in the training data are in 0–100 (validated by the ML project), but a regression model can output outside that range;
  the orchestrator shows such outputs unchanged with a warning.
* The API generates its own `request_id` and ignores incoming ones. The MCP server logs both its trace id and this `ml_request_id`.
* Not exposed by the API (so the chatbot must not claim them): accuracy metrics, feature importance, confidence intervals.
  (Notebook metrics from the earlier project — R² ≈ 0.81, MAE ≈ 3.9 — belong to that project's report, not to this API.)

## How this project maps onto it

```
StudentFeatures (mcp_server/tools/schemas.py)  — same 10 fields, same ranges, extra="forbid"
        │ validated again, then
MLApiClient.predict()  (mcp_server/ml_client.py) — POST {ML_API_BASE_URL}/predict, fixed path, no redirects
```
