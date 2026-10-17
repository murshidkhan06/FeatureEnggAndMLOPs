"""app/llm/base.py -- the provider abstraction.

The rest of the application only knows this tiny interface:  chat(messages, tools) -> reply.
Groq and Ollama are two implementations of it. Adding a third provider means adding one
class, not touching the chat logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_raw: str  # the JSON text the model produced -- NOT trusted until validated


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        """Send an OpenAI-style message list (+ tool definitions); get text and/or tool calls."""

    @abstractmethod
    async def check(self) -> dict:
        """Cheap reachability/configuration check for /health/dependencies. Never raises."""
