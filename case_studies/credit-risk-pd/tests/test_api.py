"""API contract tests.  They need models/champion.joblib (run `python -m src.models.train --bootstrap-champion` first)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import project as P
from src.api import app as api_module
from src.common.paths import CHAMPION_PATH, ROOT
from src.features.build_features import add_derived_features

pytestmark = pytest.mark.skipif(not CHAMPION_PATH.exists(), reason="train a model first")
SAMPLE = json.loads((Path(ROOT) / "configs" / "sample_request.json").read_text())


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    api_module.PREDICTION_LOG = tmp_path_factory.mktemp("logs") / "predictions.jsonl"       # keep the real log clean
    with TestClient(api_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_valid_contract(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"customer_id", "probability_of_default_6m", "risk_band", "contact_recommended", "model_version", "features_as_of"}
    assert 0.0 <= b["probability_of_default_6m"] <= 1.0 and b["risk_band"] in ("LOW", "MEDIUM", "HIGH")
    assert b["customer_id"] == SAMPLE["customer_id"] and b["features_as_of"] == SAMPLE["features_as_of"]      # timestamp echo


def test_warning_signs_score_higher_than_a_clean_record(client):
    risky = {**SAMPLE, "missed_payment_count": 3, "loan_payment_ratio": 0.5, "salary_credit_frequency": 0.5, "credit_utilization": 0.95, "balance_trend": -9000.0}
    clean = {**SAMPLE, "missed_payment_count": 0, "loan_payment_ratio": 1.0, "salary_credit_frequency": 1.0, "credit_utilization": 0.1, "balance_trend": 500.0}
    assert (client.post("/predict", json=risky).json()["probability_of_default_6m"] > client.post("/predict", json=clean).json()["probability_of_default_6m"])


def test_nullable_fields_are_accepted(client):
    r = client.post("/predict", json={**SAMPLE, "credit_utilization": None, "income_stability": None})
    assert r.status_code == 200 and 0 <= r.json()["probability_of_default_6m"] <= 1


@pytest.mark.parametrize("patch", [{"credit_utilization": 7.5}, {"missed_payment_count": 9}, {"loan_payment_ratio": 1.4}, {"salary_credit_frequency": -1},
                                   {"account_age": -3}, {"features_as_of": "yesterday"}, {"customer_id": 0}])
def test_out_of_range_values_are_rejected_with_422(client, patch):
    assert client.post("/predict", json={**SAMPLE, **patch}).status_code == 422


def test_missing_and_garbage_requests_are_422(client):
    assert client.post("/predict", json={"nonsense": 1}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in SAMPLE.items() if k != "features_as_of"}).status_code == 422


def test_customer_id_is_not_written_to_the_prediction_log(client):
    client.post("/predict", json=SAMPLE)
    rows = pd.read_json(api_module.PREDICTION_LOG, lines=True)
    assert "customer_id" not in rows.columns and set(P.DRIFT_FEATURES) <= set(rows.columns)       # drift features are logged for monitoring


def test_online_features_equal_offline_features():
    """No training-serving skew: the API builds derived features with the SAME function as training."""
    frame = P.request_to_frame(P.PredictRequest(**SAMPLE))
    base = pd.DataFrame([{k: SAMPLE[k] for k in P.DRIFT_FEATURES}]).astype(float)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), add_derived_features(base)[P.INPUT_COLUMNS].reset_index(drop=True))
    assert list(frame.columns) == P.INPUT_COLUMNS


def test_risk_bands():
    assert (P.risk_level(0.02), P.risk_level(0.05), P.risk_level(0.07), P.risk_level(0.12), P.risk_level(0.9)) == ("LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH")


def test_metrics_and_model_info(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "model_predictions_total" in text and "model_info" in text
    info = client.get("/model-info").json()
    assert info["model_version"] and "roc_auc" in info["metrics"]
