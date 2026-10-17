"""app/services/chat_service.py

THE HEART OF THE ORCHESTRATOR: one chat turn.

    user question
        -> LLM (with the MCP tool menu)               "what should we do?"
        -> LLM asks to call a tool                     (or just answers in text)
        -> CODE validates the tool arguments           required? valid? did the user say it?
              missing/unclear -> ask the user (no tool call, no model call)
        -> MCP tool runs  -> ML API -> fitted pipeline -> REAL prediction
        -> LLM writes a short explanation              "explain this result"
        -> CODE checks the explanation doesn't invent a score
        -> structured response for React

Who decides what:
    LLM   : understands language, proposes tool calls, writes the explanation.
    CODE  : enforces the rules (this file + guards.py). Never trusts the LLM blindly.
    MODEL : the only source of a predicted score.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.llm.base import LLMProvider, LLMReply, ToolCall
from app.llm.errors import LLMError
from app.logging_setup import log_event
from app.mcp_client.client import McpSession, McpUnavailableError, ToolOutcome, ToolSpec
from app.schemas.api import (
    ChatRequest, ChatResponse, ErrorInfo, FieldProblem, PredictionResult, ProfileClarification,
    ProviderInfo, ToolCallRecord,
)

from . import guards
from .prompts import SYSTEM_PROMPT
from .sessions import Conversation, ConversationStore

McpFactory = Callable[[str], McpSession]  # trace_id -> a (not yet entered) MCP session

PREDICT_TOOL = "predict_exam_score"
INFO_TOOL = "get_model_information"


class _ParsedCall:
    def __init__(self, call: ToolCall, args: dict, label: str | None):
        self.call = call
        self.args = args
        self.label = label


class ChatService:
    def __init__(self, settings: Settings, provider: LLMProvider, store: ConversationStore, mcp_factory: McpFactory):
        self._settings = settings
        self._provider = provider
        self._store = store
        self._mcp_factory = mcp_factory

    # ---------------------------------------------------------------------------------
    async def handle(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        started = time.perf_counter()
        conv = self._store.get_or_create(request.conversation_id)
        user_text = conv.user_text(request.message)  # everything the user has ever said here (for provenance)
        new_messages: list[dict] = [{"role": "user", "content": request.message}]

        predictions: list[PredictionResult] = []
        records: list[ToolCallRecord] = []
        model_info: dict | None = None
        notices: list[str] = []
        final_text = ""

        async with self._mcp_factory(trace_id) as mcp:
            tools = await mcp.list_tools()
            schemas = {t.name: t.input_schema for t in tools}
            llm_tools = [_to_llm_tool(t) for t in tools]

            for _round in range(self._settings.max_tool_rounds + 1):
                work = [{"role": "system", "content": SYSTEM_PROMPT}] + conv.messages + new_messages

                # ---- ask the LLM -----------------------------------------------------
                try:
                    reply = await self._ask_llm(work, llm_tools, _round)
                    parsed = self._parse_calls(reply, schemas, len(predictions))
                except LLMError as exc:
                    if predictions:  # we already HAVE real results: don't throw them away
                        notices.append(f"The language model failed after the prediction ({exc.code}); showing a plain summary instead.")
                        final_text = _summarise(predictions)
                        break
                    raise

                if not reply.tool_calls:
                    final_text = reply.content.strip()
                    break
                if _round >= self._settings.max_tool_rounds:
                    raise LLMError("LLM_TOO_MANY_STEPS", "The language model kept requesting tools without finishing.", 502)

                # ---- validate every requested call BEFORE running any of them ----------
                clarifications: list[ProfileClarification] = []
                for item in parsed:
                    if item.call.name != PREDICT_TOOL:
                        continue
                    issues = self._check_args(item, schemas[PREDICT_TOOL], user_text, request.message)
                    if issues:
                        clarifications.append(_to_clarification(item, issues, schemas[PREDICT_TOOL]))
                        records.append(ToolCallRecord(tool=PREDICT_TOOL, status="rejected", label=item.label,
                                                      error_code="NEEDS_CLARIFICATION"))
                if clarifications:
                    text = _clarification_text(clarifications, schemas[PREDICT_TOOL])
                    log_event("tool_call_rejected", fields_missing=sum(len(c.missing_fields) for c in clarifications),
                              fields_problem=sum(len(c.problems) for c in clarifications))
                    return self._finish(conv, request, trace_id, started, new_messages, text, kind="clarification",
                                        records=records, clarification=clarifications)

                # ---- run the tools (through MCP) ---------------------------------------
                mark = len(new_messages)
                new_messages.append(_assistant_tool_message(reply, parsed))
                for item in parsed:
                    outcome = await mcp.call_tool(item.call.name, item.args)
                    records.append(ToolCallRecord(
                        tool=item.call.name, status="ok" if outcome.ok else "error", label=item.label,
                        error_code=outcome.error_code, latency_ms=round(outcome.latency_ms, 1)))
                    log_event("tool_call", tool=item.call.name, status="ok" if outcome.ok else "error",
                              error_code=outcome.error_code, latency_ms=round(outcome.latency_ms, 1))
                    if not outcome.ok:
                        del new_messages[mark:]  # forget this half-finished round
                        err, text = _failure(outcome)
                        return self._finish(conv, request, trace_id, started, new_messages, text, kind="error",
                                            records=records, error=err)
                    if item.call.name == PREDICT_TOOL:
                        predictions.append(_to_prediction(outcome, item.label))
                    else:
                        model_info = outcome.data
                    new_messages.append({"role": "tool", "tool_call_id": item.call.id,
                                         "content": json.dumps(_for_llm(outcome.data))})

        # ---- the explanation is written; check it before we show it -------------------
        allowed = conv.scores_seen + [p.predicted_final_score for p in predictions]
        wrong = guards.find_ungrounded_claims(final_text, allowed)
        if wrong:
            log_event("assistant_text_replaced", level=logging.WARNING, reason="ungrounded_score_claim")
            notices.append("The assistant's wording contained a score the model never produced, so it was replaced by a safe summary.")
            final_text = _summarise(predictions) if predictions else (
                "I haven't run the prediction model for this, so I can't state a predicted score. "
                "Please give me the student's details and I will ask the model.")
        elif predictions and not final_text:
            final_text = _summarise(predictions)
        elif not final_text:
            final_text = "I'm not sure how to help with that. I can predict exam scores from student details - just ask."

        notices += self._range_notices(predictions)
        kind = "prediction" if predictions else ("information" if model_info else "text")
        return self._finish(conv, request, trace_id, started, new_messages, final_text, kind=kind, records=records,
                            predictions=predictions, model_info=model_info, notices=notices)

    def _range_notices(self, predictions: list[PredictionResult]) -> list[str]:
        """A regression model can output ANY number. If it is impossible for a score, say so plainly."""
        low, high = self._settings.score_min, self._settings.score_max
        return [
            f"{p.label}: the model returned {p.predicted_final_score:g}, but final scores are normally between "
            f"{low:g} and {high:g}. This usually means these inputs are unlike the data the model was trained on, "
            f"so treat this number with caution."
            for p in predictions if not low <= p.predicted_final_score <= high
        ]

    # ---------------------------------------------------------------------------------
    async def _ask_llm(self, messages: list[dict], tools: list[dict], round_no: int) -> LLMReply:
        t0 = time.perf_counter()
        try:
            reply = await self._provider.chat(messages, tools)
        except LLMError as exc:
            log_event("llm_call", level=logging.WARNING, provider=self._provider.name, model=self._provider.model,
                      round=round_no, outcome=exc.code, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            raise
        log_event("llm_call", provider=self._provider.name, model=self._provider.model, round=round_no,
                  outcome="ok", n_tool_calls=len(reply.tool_calls), latency_ms=round(reply.latency_ms, 1))
        return reply

    def _parse_calls(self, reply: LLMReply, schemas: dict[str, dict], already_predicted: int) -> list[_ParsedCall]:
        """Turn the LLM's raw tool calls into trusted-shape objects, or raise LLMError."""
        parsed: list[_ParsedCall] = []
        n_predict = sum(1 for c in reply.tool_calls if c.name == PREDICT_TOOL)
        if already_predicted + n_predict > self._settings.max_predictions_per_turn:
            raise LLMError("LLM_TOO_MANY_CALLS", f"At most {self._settings.max_predictions_per_turn} predictions are allowed per question.", 502)

        predict_index = 0
        for call in reply.tool_calls:
            if call.name not in schemas:
                raise LLMError("LLM_INVALID_TOOL_CALL", f"The language model asked for a tool that does not exist or is not allowed ('{call.name[:40]}').", 502)
            args = guards.parse_arguments(call.arguments_raw)
            if args is None:
                raise LLMError("LLM_INVALID_TOOL_ARGS", "The language model produced tool arguments that are not valid JSON. Please try rephrasing your question.", 502)
            label = None
            if call.name == PREDICT_TOOL:
                predict_index += 1
                label = "Student" if (n_predict + already_predicted) == 1 else f"Profile {already_predicted + predict_index}"
                args = guards.coerce_numeric_strings(schemas[call.name], args)
            else:
                args = {}  # get_model_information takes no arguments; ignore anything the LLM adds
            parsed.append(_ParsedCall(call, args, label))
        return parsed

    def _check_args(self, item: _ParsedCall, schema: dict, user_text: str, latest_message: str) -> list[guards.ArgIssue]:
        issues = guards.validate_arguments(schema, item.args)
        issues += guards.check_unit_mismatch(schema, item.args, latest_message)
        if self._settings.strict_provenance:
            bad_keys = {i.field for i in issues}
            candidate = {k: v for k, v in item.args.items() if k not in bad_keys and k in schema.get("properties", {})}
            issues += guards.check_provenance(candidate, user_text)
        return issues

    # ---------------------------------------------------------------------------------
    def _finish(self, conv: Conversation, request: ChatRequest, trace_id: str, started: float,
                new_messages: list[dict], text: str, *, kind: str, records: list[ToolCallRecord],
                predictions: list[PredictionResult] | None = None, clarification: list[ProfileClarification] | None = None,
                error: ErrorInfo | None = None, model_info: dict | None = None, notices: list[str] | None = None) -> ChatResponse:
        predictions = predictions or []
        conv.messages = conv.messages + new_messages + [{"role": "assistant", "content": text}]
        conv.scores_seen = (conv.scores_seen + [p.predicted_final_score for p in predictions])[-50:]
        self._store.save(conv)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        log_event("chat_turn_done", kind=kind, n_predictions=len(predictions), n_tool_calls=len(records), latency_ms=latency_ms)
        return ChatResponse(
            trace_id=trace_id, conversation_id=conv.id, kind=kind, assistant_message=text,  # type: ignore[arg-type]
            predictions=predictions, clarification=clarification, model_information=model_info,
            tool_calls=records, error=error, notices=notices or [],
            provider=ProviderInfo(name=self._provider.name, model=self._provider.model), latency_ms=latency_ms,
        )


# ---- helpers (pure functions: easy to read and to test) ----------------------------------

def _to_llm_tool(spec: ToolSpec) -> dict:
    return {"type": "function", "function": {"name": spec.name, "description": spec.description, "parameters": spec.input_schema}}


def _assistant_tool_message(reply: LLMReply, parsed: list[_ParsedCall]) -> dict:
    return {
        "role": "assistant",
        "content": reply.content or None,
        "tool_calls": [
            {"id": p.call.id, "type": "function", "function": {"name": p.call.name, "arguments": json.dumps(p.args)}}
            for p in parsed
        ],
    }


def _for_llm(data: dict) -> dict:
    """What the LLM sees of a tool result: the facts, minus plumbing ids."""
    return {k: v for k, v in data.items() if k not in ("ml_request_id", "error") and v is not None}


def _to_prediction(outcome: ToolOutcome, label: str | None) -> PredictionResult:
    d = outcome.data
    return PredictionResult(
        label=label or "Student", predicted_final_score=float(d["predicted_final_score"]), features=d.get("features_used") or {},
        model_version=d.get("model_version"), ml_request_id=d.get("ml_request_id"),
    )


def _summarise(predictions: list[PredictionResult]) -> str:
    """A plain, LLM-free sentence built straight from the model's output."""
    if len(predictions) == 1:
        return f"The model's predicted final exam score is {predictions[0].predicted_final_score:.1f}. This is the model's estimate, not a guarantee."
    lines = [f"- {p.label}: {p.predicted_final_score:.1f}" for p in predictions]
    return "The model's predicted final exam scores:\n" + "\n".join(lines) + "\nThese are the model's estimates, not guarantees."


def _to_clarification(item: _ParsedCall, issues: list[guards.ArgIssue], schema: dict) -> ProfileClarification:
    # A value the user never gave (the LLM made it up) is simply "missing" from the student's point of view.
    absent = ("missing", "ungrounded")
    flagged_fields = {i.field for i in issues}
    understood = {k: v for k, v in item.args.items() if k in schema.get("properties", {}) and k not in flagged_fields}
    return ProfileClarification(
        label=item.label or "Student",
        missing_fields=[i.field for i in issues if i.kind in absent],
        problems=[FieldProblem(field=i.field, problem=i.message) for i in issues if i.kind not in absent],
        understood_values=understood,
    )


def _clarification_text(items: list[ProfileClarification], schema: dict) -> str:
    props = schema.get("properties", {})
    blocks: list[str] = []
    for c in items:
        lines: list[str] = []
        if c.understood_values:
            got = ", ".join(f"{k} = {guards.display_value(v)}" for k, v in c.understood_values.items())
            lines.append(f"So far I have: {got}.")
        if c.missing_fields:
            lines.append("I still need:")
            lines += [f"  - {f}: {props.get(f, {}).get('description', '')}".rstrip(": ") for f in c.missing_fields]
        if c.problems:
            lines.append("These need to be checked:")
            lines += [f"  - {p.field}: {p.problem}" for p in c.problems]
        header = f"{c.label}:\n" if len(items) > 1 else ""
        blocks.append(header + "\n".join(lines))
    intro = "I can't run the prediction model yet, because I don't want to guess any values."
    return intro + "\n\n" + "\n\n".join(blocks) + "\n\nPlease send the missing or corrected values and I will run the model."


def _failure(outcome: ToolOutcome) -> tuple[ErrorInfo, str]:
    code = outcome.error_code or "TOOL_ERROR"
    err_block = (outcome.data or {}).get("error") or {}
    details = [str(d) for d in err_block.get("details", [])][:10]
    retryable = err_block.get("retryable")
    if code in ("ML_API_UNAVAILABLE", "ML_API_TIMEOUT", "MCP_UNAVAILABLE"):
        text = "I couldn't get a prediction because the prediction service is not reachable right now. No prediction was made. Please try again in a moment."
    elif code in ("INVALID_INPUT", "ML_API_REJECTED_INPUT", "MCP_TOOL_ERROR"):
        text = "The prediction was not made because some input values were rejected:\n" + "\n".join(f"  - {d}" for d in details or [outcome.error_message or "invalid input"]) + "\nPlease correct them and try again."
    elif code == "ML_API_MODEL_NOT_LOADED":
        text = "The prediction service is running but its model is not loaded yet. No prediction was made."
    else:
        text = f"The prediction service reported a problem ({code}). No prediction was made."
    return ErrorInfo(code=code, message=outcome.error_message or "The tool failed.", retryable=retryable, details=details), text
