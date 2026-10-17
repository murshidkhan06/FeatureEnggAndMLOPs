"""LLM provider configuration and the OpenAI-compatible HTTP client (network faked with MockTransport)."""
from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.llm.base import LLMReply
from app.llm.errors import LLMError
from app.llm.factory import UnconfiguredProvider, build_provider
from app.llm.openai_compat import OpenAICompatibleProvider

SECRET = "gsk_THIS_MUST_NEVER_LEAK_123"
TOOLS = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
MESSAGES = [{"role": "user", "content": "hi"}]


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# ---- configuration: switching provider is configuration only ------------------------------------

def test_groq_is_the_default_provider():
    assert settings().llm_provider == "groq"


def test_groq_provider_is_built_from_config():
    p = build_provider(settings(groq_api_key=SECRET, groq_model="my-groq-model"))
    assert isinstance(p, OpenAICompatibleProvider)
    assert (p.name, p.model) == ("groq", "my-groq-model") and p._base_url == "https://api.groq.com/openai/v1"


def test_ollama_provider_is_built_from_config_without_any_key():
    p = build_provider(settings(llm_provider="ollama", ollama_model="llama3.1:8b", ollama_base_url="http://host.docker.internal:11434/"))
    assert (p.name, p.model) == ("ollama", "llama3.1:8b")
    assert p._base_url == "http://host.docker.internal:11434/v1" and p._api_key is None


def test_model_names_are_never_hardcoded():
    assert settings().groq_model == "" and settings().ollama_model == ""
    assert build_provider(settings(groq_api_key=SECRET, groq_model="a")).model != build_provider(settings(groq_api_key=SECRET, groq_model="b")).model


@pytest.mark.parametrize("kw,fragment", [
    (dict(llm_provider="groq", groq_model="m"), "GROQ_API_KEY"),
    (dict(llm_provider="groq", groq_api_key=SECRET), "GROQ_MODEL"),
    (dict(llm_provider="ollama"), "OLLAMA_MODEL"),
])
async def test_incomplete_config_gives_a_clear_error_not_a_crash(kw, fragment):
    p = build_provider(settings(**kw))
    assert isinstance(p, UnconfiguredProvider)
    with pytest.raises(LLMError) as exc:
        await p.chat(MESSAGES, TOOLS)
    assert exc.value.code == "LLM_NOT_CONFIGURED" and exc.value.http_status == 503 and fragment in exc.value.message
    assert (await p.check())["ok"] is False


def test_unknown_provider_name_is_rejected():
    with pytest.raises(ValueError):
        settings(llm_provider="openai")


def test_settings_never_print_the_api_key():
    s = settings(groq_api_key=SECRET, groq_model="m")
    assert SECRET not in repr(s) and SECRET not in str(s) and SECRET not in s.model_dump_json()


def test_env_vars_select_the_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    p = build_provider(Settings(_env_file=None))
    assert (p.name, p.model) == ("ollama", "qwen2.5:7b")


# ---- the HTTP client ---------------------------------------------------------------------------

def provider(handler, name="groq", model="m", key=SECRET if True else None, base="https://llm.test/v1", timeout=5):
    return OpenAICompatibleProvider(name=name, base_url=base, model=model, api_key=key, timeout_seconds=timeout,
                                    transport=httpx.MockTransport(handler))


def completion(message: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message}]})


async def test_text_reply_and_request_shape():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers.get("authorization"), json.loads(request.content)
        return completion({"role": "assistant", "content": "Hello"})

    reply = await provider(handler, model="the-model").chat(MESSAGES, TOOLS)
    assert isinstance(reply, LLMReply) and reply.content == "Hello" and reply.tool_calls == []
    assert seen["url"] == "https://llm.test/v1/chat/completions" and seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["model"] == "the-model" and seen["body"]["tools"] == TOOLS and seen["body"]["tool_choice"] == "auto"
    assert seen["body"]["temperature"] == 0.0


async def test_ollama_requests_carry_no_authorization_header():
    seen = {}
    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return completion({"content": "ok"})
    await provider(handler, name="ollama", key=None).chat(MESSAGES, [])
    assert seen["auth"] is None


async def test_tool_calls_with_string_or_object_arguments_are_parsed():
    def handler(request):
        return completion({"content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "predict_exam_score", "arguments": '{"study_hours": 5}'}},
            {"function": {"name": "get_model_information", "arguments": {"x": 1}}},   # Ollama-style: an object, and no id
        ]})
    reply = await provider(handler).chat(MESSAGES, TOOLS)
    assert [(c.id, c.name) for c in reply.tool_calls] == [("a", "predict_exam_score"), ("call_1", "get_model_information")]
    assert json.loads(reply.tool_calls[0].arguments_raw) == {"study_hours": 5} and json.loads(reply.tool_calls[1].arguments_raw) == {"x": 1}


@pytest.mark.parametrize("status,code,http", [
    (401, "LLM_AUTH_FAILED", 502), (403, "LLM_AUTH_FAILED", 502), (404, "LLM_MODEL_NOT_FOUND", 502),
    (429, "LLM_RATE_LIMITED", 503), (500, "LLM_UNAVAILABLE", 502), (503, "LLM_UNAVAILABLE", 502),
])
async def test_http_errors_map_to_clear_codes_and_never_leak_the_key(status, code, http):
    p = provider(lambda r: httpx.Response(status, json={"error": {"message": f"nope {SECRET}"}}))
    with pytest.raises(LLMError) as exc:
        await p.chat(MESSAGES, TOOLS)
    assert (exc.value.code, exc.value.http_status) == (code, http)
    assert SECRET not in exc.value.message  # we never echo provider error bodies


async def test_model_without_tool_support_is_reported_clearly():
    p = provider(lambda r: httpx.Response(400, json={"error": "registry.ollama.ai/library/x does not support tools"}), name="ollama", key=None)
    with pytest.raises(LLMError) as exc:
        await p.chat(MESSAGES, TOOLS)
    assert exc.value.code == "LLM_TOOLS_UNSUPPORTED"


async def test_missing_ollama_model_tells_you_the_pull_command():
    p = provider(lambda r: httpx.Response(404, json={"error": "model not found"}), name="ollama", key=None, model="llama3.1:8b")
    with pytest.raises(LLMError) as exc:
        await p.chat(MESSAGES, TOOLS)
    assert "ollama pull llama3.1:8b" in exc.value.message


async def test_timeouts_and_connection_errors():
    def slow(request): raise httpx.ReadTimeout("t", request=request)
    def down(request): raise httpx.ConnectError("c", request=request)
    with pytest.raises(LLMError) as t:
        await provider(slow).chat(MESSAGES, TOOLS)
    assert (t.value.code, t.value.http_status) == ("LLM_TIMEOUT", 504)
    with pytest.raises(LLMError) as c:
        await provider(down, name="ollama", key=None).chat(MESSAGES, TOOLS)
    assert c.value.code == "LLM_UNAVAILABLE" and "ollama serve" in c.value.message


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": "text"}]}, {"choices": [{"message": {"tool_calls": [{"function": {}}]}}]}])
async def test_malformed_provider_responses(body):
    with pytest.raises(LLMError) as exc:
        await provider(lambda r: httpx.Response(200, json=body)).chat(MESSAGES, TOOLS)
    assert exc.value.code == "LLM_BAD_RESPONSE"


async def test_non_json_response():
    with pytest.raises(LLMError) as exc:
        await provider(lambda r: httpx.Response(200, text="<html>")).chat(MESSAGES, TOOLS)
    assert exc.value.code == "LLM_BAD_RESPONSE"


async def test_no_silent_fallback_to_another_provider():
    """If the configured provider fails we raise - we never call a different host."""
    hosts = []
    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(500)
    p = build_provider(settings(groq_api_key=SECRET, groq_model="m"), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await p.chat(MESSAGES, TOOLS)
    assert set(hosts) == {"api.groq.com"}


async def test_ollama_health_check_reports_missing_model():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "other:latest"}]})
    p = provider(handler, name="ollama", key=None, model="llama3.1:8b", base="http://localhost:11434/v1")
    check = await p.check()
    assert check["ok"] is False and "ollama pull llama3.1:8b" in check["reason"]
    ok = provider(lambda r: httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]}), name="ollama", key=None, model="llama3.1:8b", base="http://localhost:11434/v1")
    assert (await ok.check())["ok"] is True


async def test_ollama_health_check_when_ollama_is_off():
    def down(request): raise httpx.ConnectError("refused", request=request)
    check = await provider(down, name="ollama", key=None, model="m", base="http://localhost:11434/v1").check()
    assert check["ok"] is False and "not reachable" in check["reason"]
