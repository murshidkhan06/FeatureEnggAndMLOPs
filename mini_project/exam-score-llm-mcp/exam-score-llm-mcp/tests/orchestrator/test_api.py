"""The orchestrator's HTTP API: schemas, status codes, CORS, size limits, trace IDs, health."""
from __future__ import annotations

import json

import httpx
import pytest

from app.llm.errors import LLMError
from app.main import create_app
from app.mcp_client.client import McpSession
from conftest import GOOD_FEATURES, GOOD_SENTENCE
from helpers import ScriptedLLM, call_predict, http_client, make_app, make_settings, say, use_tools

ORIGIN = "http://localhost:5173"


async def post_chat(app, body, **kw):
    async with http_client(app) as client:
        return await client.post("/chat", json=body, **kw)


# ---- request validation ---------------------------------------------------------------------

@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "   "}, {"message": "x" * 1001}, {"message": 5},
                                  {"message": "hi", "conversation_id": "short"}, {"message": "hi", "conversation_id": "bad id with spaces!"}])
async def test_invalid_requests_get_422_with_a_safe_error(fake_ml, body):
    response = await post_chat(make_app(fake_ml, ScriptedLLM()), body)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "INVALID_REQUEST" and data["trace_id"]
    assert "Traceback" not in response.text


async def test_oversized_body_is_rejected_with_413(fake_ml):
    app = make_app(fake_ml, ScriptedLLM(), max_request_bytes=500)
    response = await post_chat(app, {"message": "a" * 900})
    assert response.status_code == 413 and response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


async def test_oversized_streamed_body_without_content_length_is_rejected(fake_ml):
    app = make_app(fake_ml, ScriptedLLM(), max_request_bytes=500)
    async def chunks():
        for _ in range(10):
            yield b'{"message": "' + b"a" * 100
    async with http_client(app) as client:
        response = await client.post("/chat", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


# ---- successful chat ------------------------------------------------------------------------------

async def test_chat_returns_the_documented_schema(fake_ml):
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("The model predicts 95.2."))
    response = await post_chat(make_app(fake_ml, llm), {"message": GOOD_SENTENCE})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"trace_id", "conversation_id", "kind", "assistant_message", "predictions", "clarification",
                         "model_information", "tool_calls", "error", "notices", "provider", "latency_ms"}
    assert data["kind"] == "prediction" and data["predictions"][0]["predicted_final_score"] == 95.2
    assert data["provider"] == {"name": "groq", "model": "scripted-model"}
    assert response.headers["x-trace-id"] == data["trace_id"]


async def test_trace_id_is_generated_or_taken_from_a_valid_header(fake_ml):
    app = make_app(fake_ml, ScriptedLLM(say("a"), say("b"), say("c")))
    mine = await post_chat(app, {"message": "hi"}, headers={"X-Request-ID": "react-req-12345"})
    assert mine.json()["trace_id"] == "react-req-12345"
    bad = await post_chat(app, {"message": "hi"}, headers={"X-Request-ID": "bad id\twith junk"})
    assert bad.json()["trace_id"] != "bad id\twith junk" and len(bad.json()["trace_id"]) == 16


async def test_reset_conversation_endpoint(fake_ml):
    app = make_app(fake_ml, ScriptedLLM(say("one"), say("two")))
    first = (await post_chat(app, {"message": "hello"})).json()
    async with http_client(app) as client:
        assert (await client.delete(f"/chat/{first['conversation_id']}")).status_code == 204
        assert (await client.delete("/chat/does-not-exist-1")).status_code == 204
    assert len(app.state.store) == 0


# ---- downstream failures -------------------------------------------------------------------------

@pytest.mark.parametrize("error,status", [
    (LLMError("LLM_UNAVAILABLE", "Could not connect to groq.", 502), 502),
    (LLMError("LLM_TIMEOUT", "no answer", 504), 504),
    (LLMError("LLM_RATE_LIMITED", "slow down", 503), 503),
    (LLMError("LLM_AUTH_FAILED", "bad credentials", 502), 502),
])
async def test_llm_provider_failures_become_clear_error_responses(fake_ml, error, status):
    response = await post_chat(make_app(fake_ml, ScriptedLLM(error)), {"message": "hi"})
    assert response.status_code == status
    data = response.json()
    assert data["error"]["code"] == error.code and data["error"]["message"] == error.message and data["trace_id"]


async def test_unconfigured_provider_returns_503_with_instructions(fake_ml):
    from helpers import InMemoryMcpSession
    from app.llm.factory import build_provider
    settings = make_settings(groq_api_key=None)
    app = create_app(settings, build_provider(settings), mcp_factory=lambda t: InMemoryMcpSession(fake_ml, t))
    response = await post_chat(app, {"message": "hi"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "LLM_NOT_CONFIGURED"
    assert "GROQ_API_KEY" in response.json()["error"]["message"]
    async with http_client(app) as client:           # /health still works, and explains the problem
        health = (await client.get("/health")).json()
    assert health["llm_configured"] is False and "GROQ_API_KEY" in health["llm_note"]


async def test_mcp_server_down_returns_502(fake_ml):
    settings = make_settings(mcp_server_url="http://127.0.0.1:9/mcp", mcp_timeout_seconds=2)
    app = create_app(settings, ScriptedLLM(say("never reached")))  # real McpSession -> nothing is listening on port 9
    response = await post_chat(app, {"message": "hi"})
    assert response.status_code == 502 and response.json()["error"]["code"] == "MCP_UNAVAILABLE"


async def test_ml_api_down_is_a_200_chat_response_with_an_error_object(fake_ml):
    fake_ml.mode = "down"
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)))
    response = await post_chat(make_app(fake_ml, llm), {"message": GOOD_SENTENCE})
    data = response.json()
    assert response.status_code == 200 and data["kind"] == "error" and data["error"]["code"] == "ML_API_UNAVAILABLE"
    assert data["predictions"] == []


# ---- CORS, health, secrets ---------------------------------------------------------------------

async def test_cors_allows_only_the_configured_origin(fake_ml):
    app = make_app(fake_ml, ScriptedLLM())
    async with http_client(app) as client:
        ok = await client.options("/chat", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        evil = await client.options("/chat", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-origin" not in evil.headers


async def test_health_shows_provider_and_model_but_no_secret(fake_ml):
    app = make_app(fake_ml, ScriptedLLM(), groq_api_key="gsk_TOPSECRET_VALUE")
    async with http_client(app) as client:
        response = await client.get("/health")
    assert response.status_code == 200 and "gsk_TOPSECRET_VALUE" not in response.text
    assert response.json()["status"] == "ok" and response.json()["provider"]["name"] == "groq"


async def test_dependency_health_reports_each_service(fake_ml):
    app = make_app(fake_ml, ScriptedLLM())
    async with http_client(app) as client:
        data = (await client.get("/health/dependencies")).json()
    assert data["mcp"]["ok"] is True and set(data["mcp"]["tools"]) == {"predict_exam_score", "get_model_information"}
    assert data["ml_api"]["ok"] is True and data["llm"]["ok"] is True and data["all_ok"] is True
    fake_ml.mode = "down"
    async with http_client(app) as client:
        data = (await client.get("/health/dependencies")).json()
    assert data["ml_api"]["ok"] is False and data["ml_api"]["error"] == "ML_API_UNAVAILABLE" and data["all_ok"] is False


async def test_no_secret_in_any_response_or_log(fake_ml, log_capture):
    secret = "gsk_NEVER_SHOW_THIS_9876"
    app = make_app(fake_ml, ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("95.2"), LLMError("LLM_AUTH_FAILED", "rejected", 502)), groq_api_key=secret)
    bodies = []
    for message in (GOOD_SENTENCE, "again"):
        bodies.append((await post_chat(app, {"message": message})).text)
    async with http_client(app) as client:
        bodies.append((await client.get("/health")).text)
        bodies.append((await client.get("/health/dependencies")).text)
        bodies.append((await client.get("/openapi.json")).text)
    assert secret not in "".join(bodies) and secret not in log_capture.getvalue()
    assert "chat_turn_done" in log_capture.getvalue()  # (we really did capture the logs)


async def test_logs_do_not_contain_student_values(fake_ml, log_capture):
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("95.2"))
    await post_chat(make_app(fake_ml, llm), {"message": GOOD_SENTENCE})
    text = log_capture.getvalue()
    assert "Pune" not in text and "2025-06-01" not in text and "12.5" not in text   # feature VALUES are not logged
    assert "predict_ok" in text and "chat_turn_done" in text                         # but the story is


async def test_health_tells_the_teacher_when_the_ollama_model_is_not_pulled(fake_ml):
    """Common classroom mistake: Ollama is running but `ollama pull <model>` was never done."""
    from app.llm.factory import build_provider
    from helpers import InMemoryMcpSession

    settings = make_settings(llm_provider="ollama", ollama_model="llama3.1:8b", groq_api_key=None, groq_model="")
    tags = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [{"name": "other:latest"}]}))
    app = create_app(settings, build_provider(settings, transport=tags),
                     mcp_factory=lambda trace_id: InMemoryMcpSession(fake_ml, trace_id))
    async with http_client(app) as client:
        body = (await client.get("/health")).json()
    assert body["provider"] == {"name": "ollama", "model": "llama3.1:8b"}
    assert body["llm_configured"] is False and "ollama pull llama3.1:8b" in body["llm_note"]
