"""mcp_server/ml_client.py

The ONLY code in this project that talks to the existing Exam Score Prediction API.

Safety by construction:
  * The base URL comes from configuration, never from a tool argument.
  * Only three fixed paths are ever requested: /predict, /health, /openapi.json.
  * Every call has a timeout, and every failure becomes a typed `MLApiError`
    (never a raw stack trace, never a made-up prediction).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx

PREDICT_PATH = "/predict"
HEALTH_PATH = "/health"
OPENAPI_PATH = "/openapi.json"


class MLApiError(Exception):
    """A failure talking to the ML API, with a stable `code` the orchestrator can rely on."""

    def __init__(self, code: str, message: str, *, retryable: bool, details: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or []


@dataclass
class MLPrediction:
    predicted_final_score: float
    model_version: str
    ml_request_id: str
    latency_ms: float


class MLApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0, transport: httpx.AsyncBaseTransport | None = None):
        # `transport` lets unit tests plug in a fake ML API without any network.
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            follow_redirects=False,  # never follow a redirect to somewhere unexpected
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- low-level request with uniform error mapping ----------------------------------
    async def _request(self, method: str, path: str, *, json: dict | None = None, trace_id: str | None = None) -> httpx.Response:
        headers = {"X-Request-ID": trace_id} if trace_id and trace_id != "-" else None
        try:
            return await self._client.request(method, path, json=json, headers=headers)
        except httpx.TimeoutException as exc:
            raise MLApiError("ML_API_TIMEOUT", "The prediction service took too long to respond.", retryable=True) from exc
        except httpx.ConnectError as exc:
            raise MLApiError("ML_API_UNAVAILABLE", "The prediction service is not reachable.", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise MLApiError("ML_API_UNAVAILABLE", f"Network problem talking to the prediction service ({type(exc).__name__}).", retryable=True) from exc

    # ---- public methods ----------------------------------------------------------------
    async def predict(self, payload: dict, trace_id: str | None = None) -> MLPrediction:
        started = time.perf_counter()
        response = await self._request("POST", PREDICT_PATH, json=payload, trace_id=trace_id)
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 422:
            raise MLApiError(
                "ML_API_REJECTED_INPUT",
                "The prediction service rejected the input values.",
                retryable=False,
                details=_summarise_422(response),
            )
        if response.status_code == 503:
            raise MLApiError("ML_API_MODEL_NOT_LOADED", "The prediction service is running but its model is not loaded.", retryable=True)
        if response.status_code >= 400:
            raise MLApiError("ML_API_ERROR", f"The prediction service returned HTTP {response.status_code}.", retryable=response.status_code >= 500)

        try:
            body = response.json()
            score = body["predicted_final_score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("predicted_final_score is not a finite number")
            return MLPrediction(
                predicted_final_score=float(score),
                model_version=str(body.get("model_version", "unknown")),
                ml_request_id=str(body.get("request_id", "unknown")),
                latency_ms=latency_ms,
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise MLApiError("ML_API_BAD_RESPONSE", "The prediction service returned a response in an unexpected format.", retryable=False) from exc

    async def health(self) -> dict:
        response = await self._request("GET", HEALTH_PATH)
        if response.status_code != 200:
            raise MLApiError("ML_API_ERROR", f"Health check returned HTTP {response.status_code}.", retryable=True)
        try:
            return response.json()
        except ValueError as exc:
            raise MLApiError("ML_API_BAD_RESPONSE", "Health endpoint did not return JSON.", retryable=False) from exc

    async def openapi(self) -> dict:
        response = await self._request("GET", OPENAPI_PATH)
        if response.status_code != 200:
            raise MLApiError("ML_API_ERROR", f"OpenAPI endpoint returned HTTP {response.status_code}.", retryable=True)
        try:
            return response.json()
        except ValueError as exc:
            raise MLApiError("ML_API_BAD_RESPONSE", "OpenAPI endpoint did not return JSON.", retryable=False) from exc


def _summarise_422(response: httpx.Response) -> list[str]:
    """FastAPI 422 bodies look like {"detail": [{"loc": ["body","city"], "msg": "..."}]}.
    Keep only 'field: message', capped, so errors stay short and safe to display."""
    try:
        detail = response.json().get("detail")
        if isinstance(detail, list):
            out = []
            for item in detail[:10]:
                loc = ".".join(str(p) for p in item.get("loc", []) if p != "body")
                out.append(f"{loc or 'input'}: {str(item.get('msg', 'invalid'))[:120]}")
            return out
        if isinstance(detail, str):
            return [detail[:200]]
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return []
