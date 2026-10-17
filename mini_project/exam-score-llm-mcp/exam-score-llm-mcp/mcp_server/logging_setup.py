"""mcp_server/logging_setup.py

Tiny structured (JSON-lines) logging with a trace ID.

WHY: when a classroom demo goes wrong, you want to grep ONE id and see the request travel:
    React -> orchestrator -> MCP server -> ML API
Every log line carries `trace_id`. We NEVER log secrets, and we log feature NAMES (not the
student values) to keep personal data out of logs.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

SERVICE_NAME = "mcp-server"
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_REDACT_KEYS = {"api_key", "authorization", "groq_api_key", "password", "secret", "token"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "trace_id": trace_id_var.get(),
            "event": record.getMessage(),
        }
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = "***" if key.lower() in _REDACT_KEYS else value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("exam_mcp")
    if not logger.handlers:
        # stderr, not stdout: with the stdio transport, stdout is the MCP protocol channel.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"fields": fields})
