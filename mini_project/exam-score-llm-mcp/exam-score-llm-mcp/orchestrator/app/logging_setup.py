"""app/logging_setup.py -- JSON-lines logging with a trace ID (see mcp_server/logging_setup.py
for the explanation; this is the same idea in the orchestrator)."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

SERVICE_NAME = "orchestrator"
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

# Any field whose NAME is in this set is masked, so a careless log_event(api_key=...) can't leak.
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
    logger = logging.getLogger("exam_orchestrator")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())
    # httpx logs full request URLs at INFO; keep the console readable and quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logger


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    logging.getLogger("exam_orchestrator").log(level, event, extra={"fields": fields})
