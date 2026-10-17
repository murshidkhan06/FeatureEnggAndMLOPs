"""app/llm/factory.py -- choose the provider from configuration.

Important design rule: if the configured provider is broken we say so. We NEVER silently
switch to the other one (that would hide misconfiguration and could send student data to a
cloud service the teacher did not intend).
"""
from __future__ import annotations

import httpx

from app.config import Settings

from .base import LLMProvider, LLMReply
from .errors import LLMError
from .openai_compat import OpenAICompatibleProvider


class UnconfiguredProvider(LLMProvider):
    """Stands in when configuration is incomplete. The app still starts (so /health can
    explain the problem) but every chat request gets a clear LLM_NOT_CONFIGURED error."""

    def __init__(self, name: str, model: str, reason: str):
        self.name = name
        self.model = model
        self._reason = reason

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        raise LLMError("LLM_NOT_CONFIGURED", self._reason, 503)

    async def check(self) -> dict:
        return {"ok": False, "reason": self._reason}


def build_provider(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> LLMProvider:
    if settings.llm_provider == "groq":
        key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
        if not key:
            return UnconfiguredProvider("groq", settings.groq_model, "GROQ_API_KEY is not set. Add it to orchestrator/.env (or choose LLM_PROVIDER=ollama).")
        if not settings.groq_model:
            return UnconfiguredProvider("groq", "", "GROQ_MODEL is not set. Choose a tool-calling model from the Groq console.")
        return OpenAICompatibleProvider(
            name="groq", base_url=settings.groq_base_url, model=settings.groq_model, api_key=key,
            timeout_seconds=settings.llm_timeout_seconds, temperature=settings.llm_temperature, transport=transport,
        )

    # ollama
    if not settings.ollama_model:
        return UnconfiguredProvider("ollama", "", "OLLAMA_MODEL is not set. Pull a tool-capable model (e.g. `ollama pull llama3.1:8b`) and set OLLAMA_MODEL.")
    return OpenAICompatibleProvider(
        name="ollama", base_url=f"{settings.ollama_base_url}/v1", model=settings.ollama_model, api_key=None,
        timeout_seconds=settings.llm_timeout_seconds, temperature=settings.llm_temperature, transport=transport,
    )
