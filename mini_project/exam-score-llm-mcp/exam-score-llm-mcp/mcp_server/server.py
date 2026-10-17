"""mcp_server/server.py

The MCP server: a small, typed "menu" of tools that an LLM is allowed to use.

Run it (from the mcp_server/ folder):
    python server.py

Two tools only, on purpose:
  * predict_exam_score   -> calls the existing ML API's /predict
  * get_model_information-> reports what the ML API publicly says about itself
"""
from __future__ import annotations

import uuid

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult
from starlette.requests import Request
from starlette.responses import JSONResponse

from logging_setup import configure_logging, trace_id_var
from ml_client import MLApiClient
from settings import Settings, get_settings
from tools.info import run_model_information
from tools.predict import run_prediction
from tools.schemas import (
    AttendancePct, City, EnrollmentDate, IncomeBracket, LuckyNumber, MockScore,
    ShoeSize, StudyHours,
)


def _trace_id_from(ctx: Context | None) -> str:
    """The orchestrator sends its trace ID as the HTTP header `X-Trace-Id`, so one ID
    follows the request through every service."""
    try:
        request = ctx.request_context.request  # a Starlette Request on HTTP transports
        value = request.headers.get("x-trace-id") if request is not None else None
        if value and len(value) <= 64:
            return value
    except Exception:  # noqa: BLE001 - e.g. stdio transport has no HTTP request
        pass
    return uuid.uuid4().hex[:16]


def create_server(settings: Settings | None = None, ml_client: MLApiClient | None = None) -> FastMCP:
    settings = settings or get_settings()
    logger = configure_logging(settings.log_level)
    ml = ml_client or MLApiClient(settings.ml_api_base_url, settings.ml_api_timeout_seconds)

    mcp = FastMCP(
        "exam-score-mcp",
        instructions=(
            "Tools for predicting a student's final exam score with an existing machine-learning model. "
            "The model, not the assistant, produces the number."
        ),
        host=settings.mcp_host,
        port=settings.mcp_port,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.allowed_hosts,
            allowed_origins=[],  # no browser should call this server directly
        ),
    )

    @mcp.tool(
        name="predict_exam_score",
        description=(
            "Predict a student's final exam score using the trained ML model. "
            "Call this ONLY when the user has explicitly supplied a value for EVERY field. "
            "Never guess, estimate or default a missing value -- ask the user instead. "
            "Returns the model's real prediction; do not alter it."
        ),
    )
    async def predict_exam_score(
        study_hours: StudyHours,
        attendance_pct: AttendancePct,
        mock_test_1: MockScore,
        mock_test_2: MockScore,
        mock_test_3: MockScore,
        income_bracket: IncomeBracket,
        city: City,
        enrollment_date: EnrollmentDate,
        shoe_size: ShoeSize,
        lucky_number: LuckyNumber,
        ctx: Context,
    ) -> CallToolResult:
        trace_id = _trace_id_from(ctx)
        token = trace_id_var.set(trace_id)
        try:
            return await run_prediction(
                dict(
                    study_hours=study_hours, attendance_pct=attendance_pct,
                    mock_test_1=mock_test_1, mock_test_2=mock_test_2, mock_test_3=mock_test_3,
                    income_bracket=income_bracket, city=city, enrollment_date=enrollment_date,
                    shoe_size=shoe_size, lucky_number=lucky_number,
                ),
                ml, logger, trace_id,
            )
        finally:
            trace_id_var.reset(token)

    @mcp.tool(
        name="get_model_information",
        description=(
            "Describe the prediction model's API: the input fields it expects, their types and limits, "
            "and the model version. Returns only facts published by the ML API. "
            "Does NOT include accuracy or feature importance."
        ),
    )
    async def get_model_information(ctx: Context) -> CallToolResult:
        token = trace_id_var.set(_trace_id_from(ctx))
        try:
            return await run_model_information(ml, logger)
        finally:
            trace_id_var.reset(token)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        """Liveness check used by Docker. (To check the ML API too, call get_model_information.)"""
        return JSONResponse({"status": "ok", "service": "mcp-server", "ml_api_base_url": settings.ml_api_base_url})

    return mcp


def main() -> None:
    settings = get_settings()
    server = create_server(settings)
    logger = configure_logging(settings.log_level)
    logger.info("starting", extra={"fields": {"transport": settings.mcp_transport, "ml_api_base_url": settings.ml_api_base_url,
                                              "host": settings.mcp_host, "port": settings.mcp_port}})
    server.run(transport=settings.mcp_transport)


if __name__ == "__main__":
    main()
