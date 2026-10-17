"""Test helpers for the orchestrator: a scripted fake LLM and an in-memory MCP session.

The MCP session is REAL (our actual mcp_server tools, the official SDK client) - only the
network is removed, and the ML API is replaced by `FakeMLApi`. So orchestrator tests exercise
the true tool schemas without needing Groq, Ollama, sockets, or a running ML API.
"""
from __future__ import annotations

import json

import httpx
from mcp.shared.memory import create_connected_server_and_client_session

from app.config import Settings
from app.llm.base import LLMProvider, LLMReply, ToolCall
from app.llm.errors import LLMError
from app.main import create_app
from app.mcp_client.client import McpSession
from app.services.chat_service import ChatService
from app.services.sessions import ConversationStore
from ml_client import MLApiClient
from server import create_server
from settings import Settings as McpSettings


class ScriptedLLM(LLMProvider):
    """Returns the pre-written replies in order (an exception instance is raised instead)."""

    name = "groq"
    model = "scripted-model"

    def __init__(self, *replies: LLMReply | Exception):
        self.replies = list(replies)
        self.calls: list[dict] = []  # what the orchestrator sent us each time

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        if not self.replies:
            raise AssertionError("ScriptedLLM was asked for more replies than the test provided")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def check(self) -> dict:
        return {"ok": True}


def say(text: str) -> LLMReply:
    return LLMReply(content=text)


def call_predict(args: dict, call_id: str = "call_1") -> ToolCall:
    return ToolCall(call_id, "predict_exam_score", json.dumps(args))


def use_tools(*calls: ToolCall) -> LLMReply:
    return LLMReply(content="", tool_calls=list(calls))


class InMemoryMcpSession(McpSession):
    """Same class the app uses, but connected to our real MCP server through memory streams."""

    def __init__(self, fake_ml, trace_id: str):
        super().__init__("memory://", trace_id, 5)
        self._fake_ml = fake_ml

    async def __aenter__(self):
        settings = McpSettings(ml_api_base_url="http://ml.test:8000")
        ml = MLApiClient(settings.ml_api_base_url, 2, transport=self._fake_ml.transport)
        server = create_server(settings, ml_client=ml)
        self._session = await self._stack.enter_async_context(create_connected_server_and_client_session(server))
        return self


def make_settings(**overrides) -> Settings:
    base = dict(llm_provider="groq", groq_api_key="test-key-not-real", groq_model="test-model")
    base.update(overrides)
    return Settings(_env_file=None, **base)


def make_service(fake_ml, llm: LLMProvider, **settings_overrides) -> ChatService:
    settings = make_settings(**settings_overrides)
    store = ConversationStore(settings.conversation_ttl_seconds, settings.max_conversations, settings.max_history_messages)
    return ChatService(settings, llm, store, lambda trace_id: InMemoryMcpSession(fake_ml, trace_id))


def make_app(fake_ml, llm: LLMProvider, **settings_overrides):
    settings = make_settings(**settings_overrides)
    return create_app(settings, llm, mcp_factory=lambda trace_id: InMemoryMcpSession(fake_ml, trace_id))


def http_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://orchestrator.test")


__all__ = ["ScriptedLLM", "say", "call_predict", "use_tools", "make_service", "make_app", "make_settings", "http_client", "LLMError"]
