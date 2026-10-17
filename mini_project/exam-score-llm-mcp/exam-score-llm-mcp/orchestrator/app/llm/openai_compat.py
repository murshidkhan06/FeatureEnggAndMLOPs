"""app/llm/openai_compat.py

One implementation that talks to ANY server exposing the OpenAI-style
`POST {base_url}/chat/completions` endpoint with tool calling. Both providers we support do:

    Groq   ->  https://api.groq.com/openai/v1
    Ollama ->  http://localhost:11434/v1

So "Groq vs Ollama" differs only in: base URL, API key, and model name. That is why
switching provider is pure configuration.

We use plain `httpx` (instead of a vendor SDK) so students can SEE the exact request and
response, and so unit tests can fake the network with `httpx.MockTransport`.
"""
from __future__ import annotations

import json
import time

import httpx

from .base import LLMProvider, LLMReply, ToolCall
from .errors import LLMError


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float,
        temperature: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.name = name
        self.model = model
        self._base_url = base_url
        self._api_key = api_key  # kept private; only ever placed in the Authorization header
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=httpx.Timeout(self._timeout), transport=self._transport, follow_redirects=False)

    # ------------------------------------------------------------------------------------
    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        payload: dict = {"model": self.model, "messages": messages, "temperature": self._temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        try:
            async with self._client() as client:
                response = await client.post(f"{self._base_url}/chat/completions", headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError("LLM_TIMEOUT", f"The {self.name} model did not answer within {self._timeout:.0f} seconds.", 504) from exc
        except httpx.ConnectError as exc:
            hint = " Is Ollama running? (try `ollama serve`)" if self.name == "ollama" else " Check your internet connection."
            raise LLMError("LLM_UNAVAILABLE", f"Could not connect to {self.name}.{hint}", 502) from exc
        except httpx.HTTPError as exc:
            raise LLMError("LLM_UNAVAILABLE", f"Network problem talking to {self.name} ({type(exc).__name__}).", 502) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        self._raise_for_status(response)

        try:
            body = response.json()
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            tool_calls = [
                ToolCall(
                    id=str(tc.get("id") or f"call_{i}"),
                    name=str(tc["function"]["name"]),
                    arguments_raw=_arguments_as_text(tc["function"].get("arguments")),
                )
                for i, tc in enumerate(message.get("tool_calls") or [])
            ]
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError("LLM_BAD_RESPONSE", f"The {self.name} response was not in the expected format.", 502) from exc

        return LLMReply(content=content, tool_calls=tool_calls, latency_ms=latency_ms)

    # ------------------------------------------------------------------------------------
    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        detail = _safe_error_text(response)
        if status in (401, 403):
            raise LLMError("LLM_AUTH_FAILED", f"{self.name} rejected the credentials. Check GROQ_API_KEY.", 502)
        if status == 404:
            if self.name == "ollama":
                raise LLMError("LLM_MODEL_NOT_FOUND", f"Ollama does not have model '{self.model}'. Run: ollama pull {self.model}", 502)
            raise LLMError("LLM_MODEL_NOT_FOUND", f"{self.name} could not find model '{self.model}'. Check GROQ_MODEL.", 502)
        if status == 429:
            raise LLMError("LLM_RATE_LIMITED", f"{self.name} is rate-limiting requests. Wait a moment and try again.", 503)
        if status == 400 and "tool" in detail.lower():
            raise LLMError("LLM_TOOLS_UNSUPPORTED", f"Model '{self.model}' on {self.name} does not seem to support tool calling. Choose a tool-capable model.", 502)
        raise LLMError("LLM_UNAVAILABLE", f"{self.name} returned HTTP {status}.", 502)

    async def check(self) -> dict:
        """Cheap checks only -- we do NOT spend tokens in a health check."""
        if not self.model:
            return {"ok": False, "reason": f"{'GROQ_MODEL' if self.name == 'groq' else 'OLLAMA_MODEL'} is not set"}
        if self.name == "groq":
            return {"ok": True, "note": "API key present; model reachability is checked on first use"}
        try:
            async with self._client() as client:
                response = await client.get(self._base_url.removesuffix("/v1") + "/api/tags", timeout=3)
            if response.status_code != 200:
                return {"ok": False, "reason": f"Ollama returned HTTP {response.status_code}"}
            names = {m.get("name") for m in response.json().get("models", [])}
            wanted = {self.model, f"{self.model}:latest"}
            if names.isdisjoint(wanted):
                return {"ok": False, "reason": f"model '{self.model}' is not pulled (run: ollama pull {self.model})"}
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001 - health checks must never raise
            return {"ok": False, "reason": f"Ollama not reachable ({type(exc).__name__})"}


def _arguments_as_text(arguments) -> str:
    """OpenAI/Groq send arguments as a JSON *string*; some servers send an object. Normalise to text."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def _safe_error_text(response: httpx.Response) -> str:
    try:
        return str(response.json())[:300]
    except ValueError:
        return response.text[:300]
