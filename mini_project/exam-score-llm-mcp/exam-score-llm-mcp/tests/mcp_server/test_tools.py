"""MCP server tools, exercised through a real MCP client session (in memory, no sockets)."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from conftest import GOOD_FEATURES, FakeMLApi
from ml_client import MLApiClient
from server import create_server
from settings import Settings
from tools.schemas import FEATURE_NAMES


@asynccontextmanager
async def open_session(fake_ml):
    """A real MCP client session connected in memory to our server (ML API replaced by a fake).
    (A context manager rather than a pytest fixture: the SDK uses anyio task groups, which must
    be entered and exited in the same task.)"""
    settings = Settings(ml_api_base_url="http://ml.test:8000")
    ml = MLApiClient(settings.ml_api_base_url, 2, transport=fake_ml.transport)
    server = create_server(settings, ml_client=ml)
    async with create_connected_server_and_client_session(server) as session:
        yield session


async def test_two_tools_are_exposed_with_typed_schema(fake_ml):
    async with open_session(fake_ml) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert set(tools) == {"predict_exam_score", "get_model_information"}
        schema = tools["predict_exam_score"].inputSchema
        assert set(schema["required"]) == set(FEATURE_NAMES)
        assert schema["properties"]["study_hours"]["maximum"] == 40
        assert "WEEK" in schema["properties"]["study_hours"]["description"]
        # No parameter can carry a URL, path or command:
        assert not {"url", "path", "command", "endpoint"} & set(schema["properties"])


async def test_predict_success_returns_structured_real_prediction(fake_ml):
    async with open_session(fake_ml) as session:
        fake_ml.score = 61.7
        result = await session.call_tool("predict_exam_score", GOOD_FEATURES)
        assert result.isError is False
        data = result.structuredContent
        assert data["ok"] is True and data["predicted_final_score"] == 61.7
        assert data["features_used"]["city"] == "Pune"
        assert data["model_version"] == "test-version" and data["ml_request_id"] == "ml-req-1"
        assert json.loads(result.content[0].text) == data  # text form matches structured form


async def test_missing_field_is_rejected_before_reaching_the_ml_api(fake_ml):
    async with open_session(fake_ml) as session:
        args = {k: v for k, v in GOOD_FEATURES.items() if k != "city"}
        result = await session.call_tool("predict_exam_score", args)
        assert result.isError is True and "city" in result.content[0].text
        assert fake_ml.predict_calls() == []


@pytest.mark.parametrize("field,bad", [("study_hours", 99), ("attendance_pct", -5), ("income_bracket", "Rich"), ("lucky_number", "abc")])
async def test_out_of_range_or_wrong_type_is_rejected(fake_ml, field, bad):
    async with open_session(fake_ml) as session:
        result = await session.call_tool("predict_exam_score", {**GOOD_FEATURES, field: bad})
        assert result.isError is True and field in result.content[0].text
        assert fake_ml.predict_calls() == []


async def test_semantic_validation_returns_structured_error(fake_ml):
    async with open_session(fake_ml) as session:
        result = await session.call_tool("predict_exam_score", {**GOOD_FEATURES, "enrollment_date": "2025-02-31"})
        assert result.isError is True
        err = result.structuredContent["error"]
        assert err["code"] == "INVALID_INPUT" and err["retryable"] is False
        assert any("enrollment_date" in d for d in err["details"])
        assert fake_ml.predict_calls() == []


async def test_extra_arguments_never_reach_the_ml_api(fake_ml):
    """A malicious/confused LLM adds a URL argument: it is ignored and nothing leaves via it."""
    async with open_session(fake_ml) as session:
        result = await session.call_tool("predict_exam_score", {**GOOD_FEATURES, "url": "http://evil.example/steal"})
        assert result.isError is False
        assert {r.url.host for r in fake_ml.requests} == {"ml.test"}
        assert set(json.loads(fake_ml.predict_calls()[0].content)) == set(FEATURE_NAMES)


@pytest.mark.parametrize("mode,code", [("down", "ML_API_UNAVAILABLE"), ("timeout", "ML_API_TIMEOUT"), ("reject", "ML_API_REJECTED_INPUT")])
async def test_ml_api_failures_are_structured_and_never_fake_a_prediction(fake_ml, mode, code):
    async with open_session(fake_ml) as session:
        fake_ml.mode = mode
        result = await session.call_tool("predict_exam_score", GOOD_FEATURES)
        assert result.isError is True
        data = result.structuredContent
        assert data["ok"] is False and data["predicted_final_score"] is None and data["error"]["code"] == code


async def test_model_information_reports_only_published_facts(fake_ml):
    async with open_session(fake_ml) as session:
        result = await session.call_tool("get_model_information", {})
        info = result.structuredContent
        assert info["ok"] is True and info["model_version"] == "test-version" and info["model_loaded"] is True
        names = [f["name"] for f in info["input_fields"]]
        assert names == FEATURE_NAMES and all(f["required"] for f in info["input_fields"])
        flat = json.dumps(info).lower()
        assert "r2" not in flat and "mae" not in flat  # no fabricated accuracy numbers
        assert "not exposed" in info["not_available"]


async def test_model_information_when_ml_api_down(fake_ml):
    async with open_session(fake_ml) as session:
        fake_ml.mode = "down"
        result = await session.call_tool("get_model_information", {})
        assert result.isError is True and result.structuredContent["error"]["code"] == "ML_API_UNAVAILABLE"
