"""API contract tests.  They need models/champion.joblib and data/processed/sales_history.csv
(run `python -m src.data.make_dataset` and `python -m src.models.train --bootstrap-champion` first)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import project as P
from src.api.app import app
from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, HISTORY_PATH, ROOT
from src.models.train import score

pytestmark = pytest.mark.skipif(not (CHAMPION_PATH.exists() and HISTORY_PATH.exists()), reason="generate data and train a model first")
SAMPLE = json.loads((Path(ROOT) / "configs" / "sample_request.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_model_and_history(client):
    b = client.get("/health").json()
    assert b["status"] == "ok" and b["model_loaded"] and b["history_loaded"] and b["history_last_date"] == "2025-02-28"


def test_predict_valid(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    b = r.json()
    assert b["predicted_units"] > 0 and b["lower"] <= b["predicted_units"] <= b["upper"]
    assert b["demand_band"] in ("LOW", "NORMAL", "HIGH") and b["model_version"] and b["history_stale_days"] == 0


def test_invalid_requests_are_422(client):
    assert client.post("/predict", json={"nonsense": 1}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "price": 0}).status_code == 422                 # price must be > 0
    assert client.post("/predict", json={**SAMPLE, "price": 61000}).status_code == 422             # unit error (paise) caught by the range
    assert client.post("/predict", json={**SAMPLE, "promotion": 2}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "forecast_date": "not-a-date"}).status_code == 422


def test_unknown_series_and_too_stale_history_are_422_not_500(client):
    r = client.post("/predict", json={**SAMPLE, "product_id": "P99"})
    assert r.status_code == 422 and "unknown series" in r.json()["detail"]
    r = client.post("/predict", json={**SAMPLE, "forecast_date": "2025-04-01"})
    assert r.status_code == 422 and "behind" in r.json()["detail"]


def test_missing_weather_forecast_is_allowed_with_a_warning(client):
    body = {k: v for k, v in SAMPLE.items() if k != "forecast_temp"}
    r = client.post("/predict", json=body)
    assert r.status_code == 200 and any("forecast_temp" in w for w in r.json()["warnings"])


def test_stale_history_warning(client):
    r = client.post("/predict", json={**SAMPLE, "forecast_date": "2025-03-03"})              # history ends 02-28 -> 2 days stale
    assert r.status_code == 200 and r.json()["history_stale_days"] == 2 and r.json()["warnings"]


def test_api_matches_offline_scoring(client):
    """Same features + same pipeline => the API number equals the number computed offline (no training-serving skew)."""
    art = load_artifact(CHAMPION_PATH)
    req = P.PredictRequest(**SAMPLE)
    frame = P.request_to_frame(req)
    offline = round(float(score(art["pipeline"], frame, art["input_columns"])[0]), 1)
    assert client.post("/predict", json=SAMPLE).json()["predicted_units"] == pytest.approx(offline)


def test_promotion_increases_the_forecast(client):
    base = client.post("/predict", json=SAMPLE).json()["predicted_units"]
    promo = client.post("/predict", json={**SAMPLE, "promotion": 1, "discount": 12.0, "price": 49.0}).json()["predicted_units"]
    assert promo > base


def test_model_info_and_metrics_endpoint(client):
    assert client.get("/model-info").json()["metrics"]["wape"] > 0
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "model_predictions_total" in text and "model_score_bucket" in text
