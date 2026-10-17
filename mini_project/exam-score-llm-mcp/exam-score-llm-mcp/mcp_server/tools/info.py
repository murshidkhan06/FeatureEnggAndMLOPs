"""mcp_server/tools/info.py

`get_model_information`: returns ONLY facts the ML API itself publishes
(its OpenAPI document and /health). We deliberately do NOT report accuracy, R^2,
feature importance, etc. -- the API does not expose them, so inventing them would be
fabrication. (Those numbers live in the ML project's evaluation_report.json; a lab
exercise asks students to discuss whether/how to expose them.)
"""
from __future__ import annotations

import json
import logging

from mcp.types import CallToolResult, TextContent

from logging_setup import log_event
from ml_client import MLApiClient, MLApiError

_INPUT_SCHEMA_NAME = "StudentInput"  # name of the request model in the ML API's OpenAPI doc


def _extract_fields(openapi: dict) -> list[dict]:
    schema = openapi.get("components", {}).get("schemas", {}).get(_INPUT_SCHEMA_NAME, {})
    required = set(schema.get("required", []))
    fields = []
    for name, spec in schema.get("properties", {}).items():
        item = {"name": name, "type": spec.get("type"), "required": name in required}
        for key in ("minimum", "maximum", "enum", "description"):
            if key in spec:
                item[key] = spec[key]
        fields.append(item)
    return fields


async def run_model_information(ml: MLApiClient, logger: logging.Logger) -> CallToolResult:
    try:
        openapi = await ml.openapi()
        health = await ml.health()
        info = {
            "ok": True,
            "api_title": openapi.get("info", {}).get("title"),
            "api_description": openapi.get("info", {}).get("description"),
            "api_version": openapi.get("info", {}).get("version"),
            "prediction_endpoint": "POST /predict",
            "target": "predicted_final_score (the ML API does not publish a unit or range for it)",
            "model_loaded": health.get("model_loaded"),
            "model_version": health.get("model_version"),
            "input_fields": _extract_fields(openapi),
            "not_available": "Accuracy metrics and feature importance are not exposed by the ML API.",
        }
        log_event(logger, "model_info_ok", n_fields=len(info["input_fields"]))
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(info))], structuredContent=info, isError=False)
    except MLApiError as exc:
        log_event(logger, "model_info_failed", level=logging.WARNING, code=exc.code)
        err = {"ok": False, "error": {"code": exc.code, "message": exc.message, "retryable": exc.retryable, "details": exc.details}}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(err))], structuredContent=err, isError=True)
