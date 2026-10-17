"""MCP server: input schema, ML API client, settings. No network, no LLM."""
from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from conftest import GOOD_FEATURES, FakeMLApi
from ml_client import MLApiClient, MLApiError
from settings import Settings
from tools.schemas import FEATURE_NAMES, StudentFeatures


# ---- StudentFeatures: what counts as valid input ---------------------------------------------

def test_valid_features_are_accepted():
    assert StudentFeatures(**GOOD_FEATURES).city == "Pune"


@pytest.mark.parametrize("field", FEATURE_NAMES)
def test_every_field_is_required(field):
    data = dict(GOOD_FEATURES)
    del data[field]
    with pytest.raises(ValidationError) as exc:
        StudentFeatures(**data)
    assert field in str(exc.value)


@pytest.mark.parametrize("field,bad", [
    ("study_hours", -1), ("study_hours", 41), ("attendance_pct", 101), ("mock_test_2", -0.1),
    ("income_bracket", "Rich"), ("lucky_number", 3.5), ("shoe_size", 0), ("study_hours", "lots"),
    ("enrollment_date", "01-06-2025"), ("enrollment_date", "2025-02-31"), ("city", "Pu<ne>"), ("city", ""),
])
def test_invalid_values_are_rejected(field, bad):
    with pytest.raises(ValidationError):
        StudentFeatures(**{**GOOD_FEATURES, field: bad})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        StudentFeatures(**GOOD_FEATURES, assignments_completed=8)


def test_schema_matches_the_ml_api_contract(openapi_doc):
    """Our tool inputs must stay identical to the ML API's StudentInput (saved copy of /openapi.json)."""
    api = openapi_doc["components"]["schemas"]["StudentInput"]
    ours = StudentFeatures.model_json_schema()
    assert set(ours["properties"]) == set(api["properties"])
    assert set(ours["required"]) == set(api["required"])
    for name, spec in api["properties"].items():
        mine = ours["properties"][name]
        assert mine["type"] == spec["type"], name
        for bound in ("minimum", "maximum", "enum"):
            if bound in spec:
                assert mine[bound] == spec[bound], f"{name}.{bound}"


# ---- MLApiClient: success and every failure mode ---------------------------------------------

def make_client(fake: FakeMLApi) -> MLApiClient:
    return MLApiClient("http://ml.test:8000", timeout_seconds=2, transport=fake.transport)


async def test_predict_success_returns_ml_number_unchanged(fake_ml):
    fake_ml.score = 73.4
    result = await make_client(fake_ml).predict(GOOD_FEATURES, trace_id="trace-123456")
    assert result.predicted_final_score == 73.4
    assert result.model_version == "test-version" and result.ml_request_id == "ml-req-1"
    request = fake_ml.predict_calls()[0]
    assert request.headers["x-request-id"] == "trace-123456"  # trace id is forwarded


async def test_only_fixed_destination_is_used(fake_ml):
    client = make_client(fake_ml)
    await client.predict(GOOD_FEATURES)
    await client.health()
    await client.openapi()
    assert {(r.url.host, r.url.port, r.url.path) for r in fake_ml.requests} == {
        ("ml.test", 8000, "/predict"), ("ml.test", 8000, "/health"), ("ml.test", 8000, "/openapi.json")}


@pytest.mark.parametrize("mode,code,retryable", [
    ("down", "ML_API_UNAVAILABLE", True), ("timeout", "ML_API_TIMEOUT", True), ("reject", "ML_API_REJECTED_INPUT", False),
    ("notloaded", "ML_API_MODEL_NOT_LOADED", True), ("error500", "ML_API_ERROR", True), ("garbage", "ML_API_BAD_RESPONSE", False),
])
async def test_failures_become_typed_errors(fake_ml, mode, code, retryable):
    fake_ml.mode = mode
    with pytest.raises(MLApiError) as exc:
        await make_client(fake_ml).predict(GOOD_FEATURES)
    assert exc.value.code == code and exc.value.retryable is retryable


async def test_rejected_input_error_lists_the_fields(fake_ml):
    fake_ml.mode = "reject"
    with pytest.raises(MLApiError) as exc:
        await make_client(fake_ml).predict(GOOD_FEATURES)
    assert exc.value.details == ["city: bad city"]


async def test_nan_or_non_numeric_score_is_not_trusted(fake_ml):
    for bad in ("high", None, True):
        client = MLApiClient("http://ml.test", transport=httpx.MockTransport(
            lambda r, bad=bad: httpx.Response(200, json={"predicted_final_score": bad})))
        with pytest.raises(MLApiError) as exc:
            await client.predict(GOOD_FEATURES)
        assert exc.value.code == "ML_API_BAD_RESPONSE"


# ---- Settings ---------------------------------------------------------------------------------

def test_settings_reject_non_http_urls():
    with pytest.raises(ValidationError):
        Settings(ml_api_base_url="file:///etc/passwd")
    with pytest.raises(ValidationError):
        Settings(ml_api_base_url="localhost:8000")


def test_settings_strip_trailing_slash_and_parse_hosts():
    s = Settings(ml_api_base_url="http://ml-api:8000/", mcp_allowed_hosts="a:*, b:*")
    assert s.ml_api_base_url == "http://ml-api:8000" and s.allowed_hosts == ["a:*", "b:*"]
