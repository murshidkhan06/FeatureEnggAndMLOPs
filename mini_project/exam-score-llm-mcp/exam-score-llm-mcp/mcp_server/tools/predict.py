"""mcp_server/tools/predict.py

Implementation of the `predict_exam_score` tool, kept separate from the MCP plumbing in
server.py so it is easy to unit-test and easy to read.

Flow:  validate (strict) -> call the ML API -> return a STRUCTURED result (success or error).
"""
from __future__ import annotations

import logging
import time

from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from logging_setup import log_event
from ml_client import MLApiClient, MLApiError
from tools.schemas import PredictionToolResult, StudentFeatures, ToolError


def _as_call_tool_result(result: PredictionToolResult) -> CallToolResult:
    """Wrap our result so BOTH the machine-readable form (structuredContent) and a text
    form (for clients that only read text) are present. isError tells the client whether
    the tool succeeded."""
    data = result.model_dump(mode="json")
    import json

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data))],
        structuredContent=data,
        isError=not result.ok,
    )


def _error(code: str, message: str, retryable: bool, details: list[str] | None = None) -> CallToolResult:
    return _as_call_tool_result(
        PredictionToolResult(ok=False, error=ToolError(code=code, message=message, retryable=retryable, details=details or []))
    )


async def run_prediction(raw_args: dict, ml: MLApiClient, logger: logging.Logger, trace_id: str) -> CallToolResult:
    started = time.perf_counter()

    # 1) Strict validation. Missing, wrongly-typed or out-of-range values stop here and
    #    NEVER reach the ML API.
    try:
        features = StudentFeatures(**raw_args)
    except ValidationError as exc:
        details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()][:10]
        log_event(logger, "predict_rejected_invalid_input", level=logging.WARNING, fields_invalid=[d.split(":")[0] for d in details])
        return _error("INVALID_INPUT", "One or more feature values are missing or invalid.", False, details)

    payload = features.model_dump()

    # 2) Call the existing ML API (the fitted pipeline does all preprocessing).
    try:
        prediction = await ml.predict(payload, trace_id=trace_id)
    except MLApiError as exc:
        log_event(logger, "predict_failed", level=logging.WARNING, code=exc.code,
                  latency_ms=round((time.perf_counter() - started) * 1000, 1))
        return _error(exc.code, exc.message, exc.retryable, exc.details)

    log_event(
        logger, "predict_ok",
        ml_request_id=prediction.ml_request_id, model_version=prediction.model_version,
        ml_latency_ms=round(prediction.latency_ms, 1), total_latency_ms=round((time.perf_counter() - started) * 1000, 1),
        feature_names=sorted(payload.keys()),  # names only, never the values
    )

    # 3) Return the model's number EXACTLY as the ML API produced it.
    return _as_call_tool_result(
        PredictionToolResult(
            ok=True,
            predicted_final_score=prediction.predicted_final_score,
            model_version=prediction.model_version,
            ml_request_id=prediction.ml_request_id,
            features_used=payload,
        )
    )
