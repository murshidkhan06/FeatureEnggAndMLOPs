"""app/main.py -- the orchestrator's HTTP API (what React talks to).

    POST   /chat                      ask a question (the main endpoint)
    DELETE /chat/{conversation_id}    forget a conversation ("reset")
    GET    /health                    liveness + which provider/model is configured (no secrets)
    GET    /health/dependencies       deeper check: LLM provider, MCP server, ML API

Run (from the orchestrator/ folder):
    uvicorn app.main:app --reload --port 8002
Interactive docs: http://localhost:8002/docs
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.errors import LLMError
from app.llm.factory import build_provider
from app.logging_setup import configure_logging, log_event, trace_id_var
from app.mcp_client.client import McpSession, McpUnavailableError
from app.schemas.api import ChatRequest, ChatResponse, ErrorInfo, ErrorResponse, HealthResponse, ProviderInfo
from app.services.chat_service import ChatService
from app.services.sessions import ConversationStore

_TRACE_OK = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


# ---- middleware ---------------------------------------------------------------------------

class RequestSizeLimitMiddleware:
    """Reject oversized request bodies (HTTP 413) before they are parsed. Counts bytes as
    they stream in, so it also protects against clients that don't send Content-Length."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(scope, receive, send)

        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes and not rejected:
                    rejected = True
                    await self._reject(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if not rejected:  # once we have answered 413, swallow whatever the app tries to send
                await send(message)

        await self.app(scope, limited_receive, guarded_send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        payload = ErrorResponse(trace_id=trace_id_var.get(), error=ErrorInfo(
            code="REQUEST_TOO_LARGE", message=f"Request body must be at most {self.max_bytes} bytes."))
        await JSONResponse(status_code=413, content=payload.model_dump())(scope, receive, send)


def _error_json(status: int, trace_id: str, code: str, message: str, details: list[str] | None = None) -> JSONResponse:
    payload = ErrorResponse(trace_id=trace_id, error=ErrorInfo(code=code, message=message, details=details or []))
    return JSONResponse(status_code=status, content=payload.model_dump(), headers={"X-Trace-Id": trace_id})


# ---- application factory ------------------------------------------------------------------

def create_app(settings: Settings | None = None, provider: LLMProvider | None = None,
               mcp_factory: Callable[[str], McpSession] | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    provider = provider or build_provider(settings)
    store = ConversationStore(settings.conversation_ttl_seconds, settings.max_conversations, settings.max_history_messages)

    if mcp_factory is None:  # tests inject a fake; normal runs use the real MCP client
        def mcp_factory(trace_id: str) -> McpSession:  # noqa: F811
            return McpSession(settings.mcp_server_url, trace_id, settings.mcp_timeout_seconds)

    service = ChatService(settings, provider, store, mcp_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        log_event("orchestrator_started", provider=provider.name, model=provider.model, mcp_server_url=settings.mcp_server_url)
        yield

    app = FastAPI(
        title="Exam Score Chat Orchestrator",
        description="Receives natural-language questions from the React UI, lets an LLM choose MCP tools, "
                    "and returns real predictions from the existing ML API.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings, app.state.provider, app.state.service, app.state.store = settings, provider, service, store

    # Middleware order: the LAST one added runs FIRST. CORS must wrap everything so even
    # error responses carry the CORS headers the browser needs.
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    @app.middleware("http")
    async def trace_and_log(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        trace_id = incoming if _TRACE_OK.match(incoming) else uuid.uuid4().hex[:16]
        token = trace_id_var.set(trace_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            log_event("unhandled_error", level=logging.ERROR, path=request.url.path)
            response = _error_json(500, trace_id, "INTERNAL_ERROR", "Something went wrong on the server.")
        response.headers["X-Trace-Id"] = trace_id
        log_event("http_request", method=request.method, path=request.url.path, status=response.status_code,
                  latency_ms=round((time.perf_counter() - started) * 1000, 1))
        trace_id_var.reset(token)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Trace-Id"],
    )

    # ---- error handlers: always the same JSON shape, never a stack trace -----------------
    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError):
        details = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()][:5]
        return _error_json(422, trace_id_var.get(), "INVALID_REQUEST", "The request was not valid.", details)

    @app.exception_handler(LLMError)
    async def on_llm_error(_: Request, exc: LLMError):
        log_event("llm_error", level=logging.WARNING, code=exc.code)
        return _error_json(exc.http_status, trace_id_var.get(), exc.code, exc.message)

    @app.exception_handler(McpUnavailableError)
    async def on_mcp_error(_: Request, exc: McpUnavailableError):
        log_event("mcp_unavailable", level=logging.WARNING)
        return _error_json(502, trace_id_var.get(), "MCP_UNAVAILABLE", str(exc))

    # ---- routes --------------------------------------------------------------------------
    @app.post("/chat", response_model=ChatResponse, responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
    async def chat(body: ChatRequest) -> ChatResponse:
        """Ask a question in natural language. Returns the assistant message AND the structured
        prediction(s) so the UI can show real model output separately from the LLM's wording."""
        return await service.handle(body, trace_id_var.get())

    @app.delete("/chat/{conversation_id}", status_code=204)
    async def reset(conversation_id: str):
        """Forget a conversation. Always 204, whether or not it existed."""
        store.delete(conversation_id)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        check = await provider.check()  # cheap: Groq = config only; Ollama = one local /api/tags call
        return HealthResponse(
            status="ok", provider=ProviderInfo(name=provider.name, model=provider.model or "(not set)"),
            llm_configured=bool(check.get("ok")), llm_note=check.get("reason"),
        )

    @app.get("/health/dependencies")
    async def dependencies() -> dict:
        """Deeper (slower) check. Tells a teacher WHICH service is down."""
        result: dict = {"llm": await provider.check(), "provider": provider.name, "model": provider.model}
        try:
            async with mcp_factory(trace_id_var.get()) as mcp:
                tools = await mcp.list_tools()
                result["mcp"] = {"ok": True, "tools": [t.name for t in tools]}
                info = await mcp.call_tool("get_model_information", {})
                result["ml_api"] = {"ok": info.ok, "model_version": info.data.get("model_version"),
                                    "error": info.error_code}
        except McpUnavailableError as exc:
            result["mcp"] = {"ok": False, "reason": str(exc)}
            result["ml_api"] = {"ok": None, "reason": "unknown because the MCP server is unreachable"}
        result["all_ok"] = bool(result["llm"].get("ok") and result["mcp"].get("ok") and result["ml_api"].get("ok"))
        return result

    return app


app = create_app()
