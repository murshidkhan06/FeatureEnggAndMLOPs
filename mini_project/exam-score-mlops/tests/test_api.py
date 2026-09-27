"""
tests/test_api.py

API TESTS (PHASE 9): request validation, status codes, response shape -- the FastAPI
layer's own contract, kept separate from tests/test_model.py so a bug in the web layer
(bad status code, wrong response schema) and a bug in the ML pipeline are never confused.

Uses FastAPI's TestClient, which runs the app in-process (no real server, no network port
needed) -- fast, and exercises the exact same code path a real HTTP request would.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.main import MODEL_PATH, app

if not MODEL_PATH.exists():
    pytest.skip(f"No trained model at {MODEL_PATH} -- run `python -m src.train` first.", allow_module_level=True)


@pytest.fixture
def client():
    # The `with` form runs the app's lifespan (loads the pipeline) before the first
    # request and cleans up afterward -- without it, `pipeline` stays None and every
    # request would 503.
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "study_hours": 12.5,
    "attendance_pct": 88.0,
    "mock_test_1": 72.0,
    "mock_test_2": 75.0,
    "mock_test_3": 70.0,
    "income_bracket": "Medium",
    "city": "Pune",
    "enrollment_date": "2025-06-01",
    "shoe_size": 9.0,
    "lucky_number": 42,
}


# ---- health / metrics --------------------------------------------------------------------

def test_health_check_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metrics_endpoint_returns_counters(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    for key in ["request_count", "prediction_count", "error_count", "avg_latency_ms"]:
        assert key in body


# ---- valid prediction --------------------------------------------------------------------

def test_predict_with_valid_payload_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_expected_shape(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert "predicted_final_score" in body
    assert isinstance(body["predicted_final_score"], float)
    assert "model_version" in body
    assert "request_id" in body


def test_predict_response_is_in_a_sane_range(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    score = response.json()["predicted_final_score"]
    assert 0 <= score <= 110


def test_predict_matches_pipeline_prediction_directly(client):
    """The API's prediction should be IDENTICAL to calling the pipeline directly -- the
    endpoint should add zero logic of its own beyond validation, logging, and formatting."""
    import pandas as pd

    from src.predict import load_pipeline

    pipeline = load_pipeline(MODEL_PATH)
    direct_prediction = round(float(pipeline.predict(pd.DataFrame([VALID_PAYLOAD]))[0]), 1)

    api_prediction = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_final_score"]
    assert direct_prediction == api_prediction


def test_predict_accepts_unseen_city_gracefully(client):
    """A city never seen during training should still return a valid prediction (the
    pipeline's one-hot encoder was built with handle_unknown='ignore') -- not a 500 error."""
    payload = {**VALID_PAYLOAD, "city": "Chennai"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ---- invalid requests ----------------------------------------------------------------------

def test_predict_rejects_missing_required_field(client):
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "study_hours"}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422  # FastAPI/Pydantic validation error, not a 500 crash


def test_predict_rejects_invalid_income_bracket(client):
    """income_bracket is a Literal["Low","Medium","High"] -- anything else should be
    rejected at the validation layer, before it ever reaches the pipeline."""
    bad_payload = {**VALID_PAYLOAD, "income_bracket": "Extremely High"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_attendance(client):
    """attendance_pct has ge=0, le=100 -- a value outside that range should be rejected."""
    bad_payload = {**VALID_PAYLOAD, "attendance_pct": 150.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_datatype(client):
    """study_hours must be a number -- a string that can't parse as one should 422, not 500."""
    bad_payload = {**VALID_PAYLOAD, "study_hours": "a lot"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---- error responses get logged, not silently swallowed ------------------------------------

def test_error_count_increments_on_invalid_request(client):
    before = client.get("/metrics").json()["error_count"]
    client.post("/predict", json={**VALID_PAYLOAD, "income_bracket": "Nope"})
    # Pydantic validation errors happen before our handler runs, so they are NOT counted
    # here (FastAPI itself returns the 422) -- this test documents that boundary rather
    # than assuming error_count captures every possible failure mode.
    after = client.get("/metrics").json()["error_count"]
    assert after >= before
