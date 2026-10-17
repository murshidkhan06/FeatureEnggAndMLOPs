"""INTEGRATION TESTS - these talk to a REAL running ML API (your exam-score-mlops service).

    # terminal 1 (in the ML API project):  uvicorn api.main:app --port 8000
    # terminal 2 (in this project):        pytest tests/integration -v

They are skipped automatically when the ML API is not reachable, so `pytest` never fails
just because you did not start it. Set ML_API_BASE_URL if it runs somewhere else.

What is real here:  the ML API + fitted pipeline, the MCP server, the MCP SDK client, the
orchestrator code.   What is scripted:  the LLM (so no Groq key / Ollama model is needed).
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from conftest import GOOD_FEATURES, GOOD_SENTENCE, ROOT
from helpers import ScriptedLLM, call_predict, http_client, make_settings, say, use_tools
from ml_client import MLApiClient
from server import create_server
from settings import Settings as McpSettings
from tools.schemas import StudentFeatures

pytestmark = pytest.mark.integration

ML_URL = os.environ.get("ML_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _ml_api_is_up() -> bool:
    try:
        return httpx.get(f"{ML_URL}/health", timeout=2).json().get("model_loaded") is True
    except Exception:  # noqa: BLE001
        return False


if not _ml_api_is_up():
    pytest.skip(f"No running ML API with a loaded model at {ML_URL} (start it, or set ML_API_BASE_URL)", allow_module_level=True)


def ml_direct(features: dict) -> dict:
    """Ask the ML API directly, bypassing everything we built - the ground truth."""
    return httpx.post(f"{ML_URL}/predict", json=features, timeout=10).json()


# ---- contract: our tool schema vs the LIVE ML API ---------------------------------------------

def test_tool_schema_matches_the_live_ml_api_contract():
    api = httpx.get(f"{ML_URL}/openapi.json", timeout=5).json()["components"]["schemas"]["StudentInput"]
    ours = StudentFeatures.model_json_schema()
    assert set(ours["properties"]) == set(api["properties"]), "field names differ from the live ML API"
    assert set(ours["required"]) == set(api["required"])
    for name, spec in api["properties"].items():
        assert ours["properties"][name]["type"] == spec["type"], name
        for bound in ("minimum", "maximum", "enum"):
            if bound in spec:
                assert ours["properties"][name][bound] == spec[bound], f"{name}.{bound}"


# ---- MCP server -> real ML API ---------------------------------------------------------------

async def test_mcp_tool_returns_exactly_what_the_ml_api_returns():
    server = create_server(McpSettings(ml_api_base_url=ML_URL))
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("predict_exam_score", GOOD_FEATURES)
    truth = ml_direct(GOOD_FEATURES)
    assert result.isError is False
    assert result.structuredContent["predicted_final_score"] == truth["predicted_final_score"]
    assert result.structuredContent["model_version"] == truth["model_version"]


async def test_mcp_model_information_reflects_the_live_api():
    server = create_server(McpSettings(ml_api_base_url=ML_URL))
    async with create_connected_server_and_client_session(server) as session:
        info = (await session.call_tool("get_model_information", {})).structuredContent
    assert info["ok"] and info["model_loaded"] and len(info["input_fields"]) == 10
    assert info["api_title"] == "Student Exam Score Predictor"


async def test_mcp_reports_unavailable_when_pointed_at_a_dead_port():
    server = create_server(McpSettings(ml_api_base_url="http://127.0.0.1:9", ml_api_timeout_seconds=2))
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("predict_exam_score", GOOD_FEATURES)
    assert result.isError and result.structuredContent["error"]["code"] == "ML_API_UNAVAILABLE"


# ---- the whole chain over real HTTP: orchestrator -> MCP server (subprocess) -> ML API -----------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class McpProcess:
    """A real MCP server running as a separate OS process, with its log captured to a file."""

    def __init__(self, url: str, log_path: Path):
        self.url, self.log_path = url, log_path

    def log_text(self) -> str:
        return self.log_path.read_text()


@pytest.fixture(scope="module")
def mcp_server_process(tmp_path_factory):
    port = _free_port()
    log_path = tmp_path_factory.mktemp("mcp") / "mcp_server.log"
    env = {**os.environ, "ML_API_BASE_URL": ML_URL, "MCP_PORT": str(port), "MCP_HOST": "127.0.0.1", "MCP_TRANSPORT": "streamable-http"}
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "server.py"], cwd=ROOT / "mcp_server", env=env, stdout=log, stderr=log, text=True)
        try:
            for _ in range(50):
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("MCP server did not start: " + log_path.read_text()[-500:])
            yield McpProcess(f"http://127.0.0.1:{port}", log_path)
        finally:
            proc.terminate()
            proc.wait(timeout=10)


async def test_full_chain_over_http_returns_the_real_prediction(mcp_server_process):
    from app.main import create_app
    settings = make_settings(mcp_server_url=f"{mcp_server_process.url}/mcp")
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("The model predicts 95.2 for this student."))
    app = create_app(settings, llm)  # REAL McpSession over HTTP
    async with http_client(app) as client:
        response = await client.post("/chat", json={"message": GOOD_SENTENCE}, headers={"X-Request-ID": "integration-trace-01"})
    data = response.json()
    truth = ml_direct(GOOD_FEATURES)["predicted_final_score"]
    assert response.status_code == 200 and data["kind"] == "prediction"
    assert data["predictions"][0]["predicted_final_score"] == truth
    assert data["trace_id"] == "integration-trace-01"


async def test_full_chain_clarification_never_reaches_the_ml_api(mcp_server_process):
    from app.main import create_app
    settings = make_settings(mcp_server_url=f"{mcp_server_process.url}/mcp")
    llm = ScriptedLLM(use_tools(call_predict({"study_hours": 10, "attendance_pct": 80})))
    app = create_app(settings, llm)
    before = httpx.get(f"{ML_URL}/metrics").json().get("prediction_count")
    async with http_client(app) as client:
        data = (await client.post("/chat", json={"message": "10 hours a week and 80% attendance"})).json()
    assert data["kind"] == "clarification"
    assert httpx.get(f"{ML_URL}/metrics").json().get("prediction_count") == before


async def test_trace_id_appears_in_the_mcp_server_log(mcp_server_process):
    """The orchestrator's trace id travels in an HTTP header to the MCP server PROCESS."""
    from app.main import create_app
    settings = make_settings(mcp_server_url=f"{mcp_server_process.url}/mcp")
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("95.2"))
    async with http_client(create_app(settings, llm)) as client:
        await client.post("/chat", json={"message": GOOD_SENTENCE}, headers={"X-Request-ID": "trace-in-mcp-log-1"})
    lines = [l for l in mcp_server_process.log_text().splitlines() if '"trace_id": "trace-in-mcp-log-1"' in l]
    assert any('"event": "predict_ok"' in l for l in lines), "MCP server never logged the orchestrator's trace id"
    ml_request_id = next(__import__("json").loads(l)["ml_request_id"] for l in lines if "predict_ok" in l)
    assert ml_request_id and ml_request_id != "unknown"   # ...and it recorded the ML API's own request id too
