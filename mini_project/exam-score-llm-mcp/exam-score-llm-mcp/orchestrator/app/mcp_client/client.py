"""app/mcp_client/client.py

A small wrapper around the official MCP Python SDK client.

Per chat turn we open a short-lived MCP session:   connect -> initialize -> list tools ->
call tools -> close.  That is simple and robust (if the MCP server restarts, the next
question just works). A production system would keep a pooled, long-lived session.

The trace ID travels to the MCP server in the `X-Trace-Id` HTTP header.
"""
from __future__ import annotations

import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.logging_setup import log_event

# Only these tools may ever be offered to the LLM, even if the MCP server later grows more.
ALLOWED_TOOLS = {"predict_exam_score", "get_model_information"}


class McpUnavailableError(Exception):
    """The MCP server could not be reached or did not answer in time."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolOutcome:
    ok: bool
    data: dict            # structured result from the tool (always a dict)
    error_code: str | None
    error_message: str | None
    latency_ms: float


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """The SDK uses task groups, so connection failures arrive wrapped in an ExceptionGroup.
    Dig out the real errors so we can classify them."""
    if isinstance(exc, BaseExceptionGroup):
        out: list[BaseException] = []
        for inner in exc.exceptions:
            out.extend(_leaf_exceptions(inner))
        return out
    return [exc]


class McpSession:
    def __init__(self, url: str, trace_id: str, timeout_seconds: float):
        self._url = url
        self._trace_id = trace_id
        self._timeout = timeout_seconds
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "McpSession":
        try:
            # NOTE: we do not wrap the context-manager entry in `anyio.fail_after`: the SDK's
            # transports use task groups, and entering a task group inside a cancel scope that
            # we then leave before exiting it breaks anyio's nesting rules. Timeouts come from
            # the HTTP client's own timeout and the session's read timeout instead.
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self._url, headers={"X-Trace-Id": self._trace_id}, timeout=self._timeout)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self._timeout))
            )
            with anyio.fail_after(self._timeout):  # safe here: nothing is entered inside this block
                await self._session.initialize()
        except BaseException as exc:  # noqa: BLE001
            await self._safe_close()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            log_event("mcp_connect_failed", level=logging.WARNING, error_types=sorted({type(e).__name__ for e in _leaf_exceptions(exc)}))
            raise McpUnavailableError(_describe(exc)) from None
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._safe_close()

    async def _safe_close(self) -> None:
        try:
            await self._stack.aclose()
        except BaseException:  # noqa: BLE001 - closing must never hide the real result
            pass

    async def list_tools(self) -> list[ToolSpec]:
        assert self._session is not None
        try:
            result = await self._session.list_tools()
        except BaseException as exc:  # noqa: BLE001
            raise McpUnavailableError(_describe(exc)) from None
        return [
            ToolSpec(t.name, t.description or "", dict(t.inputSchema))
            for t in result.tools
            if t.name in ALLOWED_TOOLS
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        assert self._session is not None
        if name not in ALLOWED_TOOLS:  # defence in depth: never forward unknown tool names
            return ToolOutcome(False, {}, "TOOL_NOT_ALLOWED", f"Tool '{name}' is not allowed.", 0.0)
        started = time.perf_counter()
        try:
            result = await self._session.call_tool(name, arguments, read_timeout_seconds=timedelta(seconds=self._timeout))
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return ToolOutcome(False, {}, "MCP_UNAVAILABLE", _describe(exc), (time.perf_counter() - started) * 1000)
        latency_ms = (time.perf_counter() - started) * 1000

        data = result.structuredContent
        if isinstance(data, dict):
            if result.isError or data.get("ok") is False:
                err = data.get("error") or {}
                return ToolOutcome(False, data, err.get("code", "TOOL_ERROR"), err.get("message", "The tool reported an error."), latency_ms)
            return ToolOutcome(True, data, None, None, latency_ms)

        # No structured content: an SDK-level failure (e.g. argument validation) comes back as text.
        text = " ".join(getattr(c, "text", "") for c in result.content)[:300]
        if result.isError:
            return ToolOutcome(False, {}, "MCP_TOOL_ERROR", text or "The tool reported an error.", latency_ms)
        return ToolOutcome(False, {}, "MCP_BAD_RESPONSE", "The tool returned no structured result.", latency_ms)


def _describe(exc: BaseException) -> str:
    leaves = _leaf_exceptions(exc)
    names = ", ".join(sorted({type(e).__name__ for e in leaves}))
    if any(isinstance(e, TimeoutError) for e in leaves):
        return "The tool server (MCP) did not answer in time."
    return f"The tool server (MCP) is not reachable ({names})."
