# `ml_api/` — the existing ML API (not copied, not modified)

The ML API is the **`exam-score-mlops`** project you built earlier: a FastAPI app serving a fitted scikit-learn
`Pipeline` (imputation → scaling → encoding → PCA → feature selection → regressor), with `POST /predict`.

This folder deliberately contains **no code**. Why?

* **One source of truth.** A copy would drift from the real model.
* **The lesson.** In production the model team and the application team are different people. The app depends on a
  *contract* (see [`../docs/ML_API_CONTRACT.md`](../docs/ML_API_CONTRACT.md)), not on the model's internals.
* **No retraining.** Nothing here touches `src/train.py`, `models/` or `params.yaml`.

## How the other services find it

| How you run | Where the ML API comes from | Address the MCP server uses |
|---|---|---|
| Docker Compose | Built from the existing project's own `Dockerfile` using the build context `ML_API_DIR` (default `../exam-score-mlops`, set in `.env`). | `http://ml-api:8000` |
| Individual terminals | `uvicorn api.main:app --port 8000` inside `exam-score-mlops` | `http://localhost:8000` |

Prerequisite: `exam-score-mlops/models/exam_score_pipeline.joblib` must exist (otherwise the container/app fails fast at
startup — run `python -m src.train` in that project first). Check with `curl http://localhost:8000/health` →
`"model_loaded": true`.

## If your ML project lives elsewhere

Set `ML_API_DIR=/absolute/or/relative/path/to/exam-score-mlops` in `exam-score-llm-mcp/.env`.
