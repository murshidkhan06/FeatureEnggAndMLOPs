"""API contract tests.  They need models/champion.joblib (run `python -m src.models.train --bootstrap-champion` first)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.common.paths import CHAMPION_PATH, ROOT

pytestmark = pytest.mark.skipif(not CHAMPION_PATH.exists(), reason="train a model first")
SAMPLE = json.loads((Path(ROOT) / "configs" / "sample_request.json").read_text())
SAMPLES = json.loads((Path(ROOT) / "configs" / "sample_requests.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_valid_has_full_response_contract(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"probability_of_default", "risk_band", "decision_hint", "reason_codes", "model_version"}
    assert 0.0 <= b["probability_of_default"] <= 1.0 and b["model_version"]
    assert b["risk_band"] in ("LOW", "MEDIUM", "HIGH")
    assert b["decision_hint"] == {"LOW": "APPROVE", "MEDIUM": "REFER_TO_UNDERWRITER", "HIGH": "DECLINE"}[b["risk_band"]]
    assert 1 <= len(b["reason_codes"]) <= 3
    assert all({"feature", "reason", "contribution"} <= set(c) for c in b["reason_codes"])


def test_three_sample_applicants_cover_the_three_bands(client):
    bands = {k: client.post("/predict", json=SAMPLES[k]).json()["risk_band"] for k in ("low_risk", "medium_risk", "high_risk")}
    assert bands == {"low_risk": "LOW", "medium_risk": "MEDIUM", "high_risk": "HIGH"}


def test_missing_annual_income_is_allowed_and_scored(client):
    r = client.post("/predict", json=SAMPLES["missing_income"])
    assert "annual_income" not in SAMPLES["missing_income"] and r.status_code == 200
    assert client.post("/predict", json={**SAMPLES["missing_income"], "annual_income": None}).json() == r.json()    # null == omitted


def test_optional_bureau_score_for_thin_file(client):
    body = {k: v for k, v in SAMPLE.items() if k != "bureau_score"}
    assert client.post("/predict", json=body).status_code == 200


@pytest.mark.parametrize("patch", [
    {"monthly_income": -5}, {"monthly_income": 0}, {"loan_term_months": 500}, {"bureau_score": 1200}, {"employment_years": -1},
    {"employment_type": "Salaried"}, {"location": "Atlantis"}, {"education": "phd"}, {"loan_amount": 1},
])
def test_out_of_range_or_unknown_values_are_422(client, patch):
    assert client.post("/predict", json={**SAMPLE, **patch}).status_code == 422


@pytest.mark.parametrize("extra", [{"gender": "female"}, {"age": 31}, {"interest_rate_offered": 11.5}, {"marital_status": "single"}])
def test_sensitive_or_leaky_fields_are_rejected_by_the_contract(client, extra):
    """extra='forbid': a sensitive / leaky column cannot sneak in through an upstream schema change."""
    assert client.post("/predict", json={**SAMPLE, **extra}).status_code == 422


def test_missing_required_field_is_422(client):
    body = {k: v for k, v in SAMPLE.items() if k != "monthly_debt"}
    assert client.post("/predict", json=body).status_code == 422
    assert client.post("/predict", json={"nonsense": 1}).status_code == 422


def test_model_info(client):
    j = client.get("/model-info").json()
    assert j["model_version"] and "roc_auc" in j["metrics"] and "log_income" in j["input_columns"]


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "model_predictions_total" in text and "model_score_bucket" in text
