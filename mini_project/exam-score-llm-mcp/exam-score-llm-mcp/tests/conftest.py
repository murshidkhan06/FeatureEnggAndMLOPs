"""tests/conftest.py -- shared test setup.

Makes the two services importable (mcp_server/ and orchestrator/ each become a source root,
exactly as when you run them from inside their own folder) and provides fakes so unit tests
never need Groq, Ollama or a running ML API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("mcp_server", "orchestrator"):
    sys.path.insert(0, str(ROOT / sub))

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# A complete, valid request for the ML API (same example the ML project's README uses).
GOOD_FEATURES = {
    "study_hours": 12.5, "attendance_pct": 88.0,
    "mock_test_1": 72.0, "mock_test_2": 75.0, "mock_test_3": 70.0,
    "income_bracket": "Medium", "city": "Pune", "enrollment_date": "2025-06-01",
    "shoe_size": 9.0, "lucky_number": 42,
}

# The same facts written the way a student would type them (used to satisfy the provenance check).
GOOD_SENTENCE = (
    "Predict for a student who studies 12.5 hours a week, attendance 88%, mock tests 72, 75 and 70, "
    "Medium income bracket, city Pune, enrolled on 2025-06-01, shoe size 9, lucky number 42."
)


@pytest.fixture
def openapi_doc() -> dict:
    """A saved copy of the REAL ML API's /openapi.json (see tests/fixtures)."""
    return json.loads((FIXTURES / "ml_api_openapi.json").read_text())


class FakeMLApi:
    """A stand-in for the ML API, as an httpx transport. Records every request it receives."""

    def __init__(self, openapi: dict, score: float = 95.2):
        self.openapi = openapi
        self.score = score
        self.requests: list[httpx.Request] = []
        self.mode = "ok"  # ok | down | timeout | reject | notloaded | error500 | garbage

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.mode == "down":
            raise httpx.ConnectError("connection refused", request=request)
        if self.mode == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        path = request.url.path
        if path == "/openapi.json":
            return httpx.Response(200, json=self.openapi)
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_loaded": True, "model_version": "test-version"})
        if path == "/predict":
            if self.mode == "reject":
                return httpx.Response(422, json={"detail": [{"loc": ["body", "city"], "msg": "bad city", "type": "x"}]})
            if self.mode == "notloaded":
                return httpx.Response(503, json={"detail": "Model is not loaded yet."})
            if self.mode == "error500":
                return httpx.Response(500, text="boom")
            if self.mode == "garbage":
                return httpx.Response(200, json={"unexpected": True})
            return httpx.Response(200, json={"predicted_final_score": self.score, "model_version": "test-version", "request_id": "ml-req-1"})
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def predict_calls(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == "/predict"]


@pytest.fixture
def fake_ml(openapi_doc) -> FakeMLApi:
    return FakeMLApi(openapi_doc)


@pytest.fixture
def log_capture():
    """Collects every JSON log line from BOTH services (orchestrator + MCP server) in one string."""
    import io
    import logging

    from app.logging_setup import JsonFormatter as OrchestratorFormatter
    from app.logging_setup import configure_logging as configure_orchestrator_logging
    from logging_setup import JsonFormatter as McpFormatter
    from logging_setup import configure_logging as configure_mcp_logging

    stream = io.StringIO()
    attached = []
    for logger, formatter in ((configure_orchestrator_logging("INFO"), OrchestratorFormatter()), (configure_mcp_logging("INFO"), McpFormatter())):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        attached.append((logger, handler))
    yield stream
    for logger, handler in attached:
        logger.removeHandler(handler)
