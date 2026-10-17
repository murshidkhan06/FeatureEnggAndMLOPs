"""Pure-function safety checks (guards.py), conversation memory, and log redaction."""
from __future__ import annotations

import json
import logging

import pytest

from app.logging_setup import JsonFormatter, log_event, configure_logging
from app.services import guards
from app.services.sessions import ConversationStore

SCHEMA = {
    "type": "object",
    "properties": {
        "study_hours": {"type": "number", "minimum": 0, "maximum": 40, "description": "Study hours per WEEK (not per day)."},
        "city": {"type": "string"},
        "income_bracket": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "lucky_number": {"type": "integer"},
        "enrollment_date": {"type": "string"},
    },
    "required": ["study_hours", "city", "income_bracket", "lucky_number", "enrollment_date"],
}


# ---- parse / coerce -----------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [('{"a": 1}', {"a": 1}), ("", {}), ("   ", {}), ("{bad", None), ("[1,2]", None), ('"text"', None)])
def test_parse_arguments(raw, expected):
    assert guards.parse_arguments(raw) == expected


def test_coerce_only_clean_numeric_strings():
    out = guards.coerce_numeric_strings(SCHEMA, {"study_hours": " 12.5 ", "lucky_number": "42", "city": "7", "income_bracket": "High"})
    assert out["study_hours"] == 12.5 and out["lucky_number"] == 42 and out["city"] == "7"
    assert guards.coerce_numeric_strings(SCHEMA, {"study_hours": "12 hours"})["study_hours"] == "12 hours"


# ---- schema validation ---------------------------------------------------------------------

def kinds(issues):
    return {(i.field, i.kind) for i in issues}


def test_validation_reports_missing_unknown_and_invalid():
    issues = guards.validate_arguments(SCHEMA, {"study_hours": 99, "income_bracket": "Rich", "pet": "cat"})
    assert kinds(issues) >= {("city", "missing"), ("lucky_number", "missing"), ("enrollment_date", "missing"),
                             ("pet", "unknown"), ("study_hours", "invalid"), ("income_bracket", "invalid")}


def test_validation_treats_null_and_blank_as_missing():
    issues = guards.validate_arguments(SCHEMA, {"study_hours": None, "city": "  ", "income_bracket": "Low", "lucky_number": 1, "enrollment_date": "2025-01-01"})
    assert kinds(issues) == {("study_hours", "missing"), ("city", "missing")}


def test_validation_rejects_booleans_as_numbers():
    issues = guards.validate_arguments(SCHEMA, {"study_hours": True, "city": "Pune", "income_bracket": "Low", "lucky_number": 1, "enrollment_date": "2025-01-01"})
    assert kinds(issues) == {("study_hours", "invalid")}


def test_complete_valid_arguments_have_no_issues():
    ok = {"study_hours": 10, "city": "Pune", "income_bracket": "Low", "lucky_number": 3, "enrollment_date": "2025-01-01"}
    assert guards.validate_arguments(SCHEMA, ok) == []


# ---- provenance ------------------------------------------------------------------------------

def test_provenance_accepts_values_the_user_wrote():
    args = {"study_hours": 12.5, "city": "Pune", "income_bracket": "Medium", "lucky_number": 42}
    text = "Student studies 12.5 hours a week, lives in pune, medium income, lucky number 42"
    assert guards.check_provenance(args, text) == []


def test_provenance_flags_values_the_user_never_wrote():
    issues = guards.check_provenance({"study_hours": 42, "lucky_number": 7, "city": "Delhi"}, "6 hours daily in Pune, lucky 8")
    assert {i.field for i in issues} == {"study_hours", "lucky_number", "city"} and all(i.kind == "ungrounded" for i in issues)


def test_provenance_matches_whole_words_only():
    assert guards.check_provenance({"income_bracket": "Low"}, "follows the slowest schedule")  # 'Low' hides inside 'slowest'
    assert guards.check_provenance({"income_bracket": "Low"}, "Low income") == []


def test_provenance_for_dates_needs_year_and_day():
    assert guards.check_provenance({"enrollment_date": "2025-06-01"}, "enrolled on 1 June 2025") == []
    assert guards.check_provenance({"enrollment_date": "2025-06-01"}, "enrolled 2025-06-01") == []
    assert guards.check_provenance({"enrollment_date": "2025-06-01"}, "enrolled in June") != []


@pytest.mark.parametrize("field,value,text", [
    ("shoe_size", 8, "studied 6 hours, completed 8 assignments"),      # number present, topic never mentioned
    ("lucky_number", 7, "I have 7 siblings"),
    ("mock_test_1", 70, "70 percent of my friends passed"),
    ("attendance_pct", 85, "I read 85 pages"),
    ("enrollment_date", "2025-06-01", "the number 2025 and 1"),
])
def test_a_number_needs_its_topic_to_be_mentioned(field, value, text):
    issues = guards.check_provenance({field: value}, text)
    assert [(i.field, i.kind) for i in issues] == [(field, "ungrounded")]
    assert str(value) not in issues[0].message                      # we never echo the LLM's invented value


@pytest.mark.parametrize("field,value,text", [
    ("shoe_size", 8, "my shoe size is 8"), ("lucky_number", 7, "lucky number: 7"), ("mock_test_2", 70, "mock test 2 was 70"),
    ("attendance_pct", 85, "attended 85% of classes"), ("study_hours", 6, "studies 6 hours"),
    ("enrollment_date", "2025-06-01", "enrolled on 1 June 2025"),
])
def test_a_number_with_its_topic_mentioned_is_accepted(field, value, text):
    assert guards.check_provenance({field: value}, text) == []


def test_provenance_never_accepts_booleans():
    assert guards.check_provenance({"flag": True}, "true 1 yes") != []


# ---- unit mismatch ----------------------------------------------------------------------------

@pytest.mark.parametrize("message,flagged", [
    ("I study 6 hours daily", True), ("6 hours per day", True), ("studies 3 hours a day", True), ("6 hours/day", True),
    ("40 hours per week", False), ("6 hours daily which is 36 hours weekly", False), ("12.5 hours", False),
])
def test_unit_mismatch_detects_daily_hours(message, flagged):
    issues = guards.check_unit_mismatch(SCHEMA, {"study_hours": 6}, message)
    assert bool(issues) is flagged
    if flagged:
        assert issues[0].field == "study_hours" and "WEEK" in issues[0].message


def test_unit_check_ignores_calls_without_study_hours():
    assert guards.check_unit_mismatch(SCHEMA, {"city": "Pune"}, "6 hours daily") == []


# ---- output guard -----------------------------------------------------------------------------

@pytest.mark.parametrize("text,allowed,bad", [
    ("The predicted final score is 95.2.", [95.2], []),
    ("The predicted final score is 95.", [95.2], []),               # rounding within 0.05 is tolerated
    ("The predicted final score is 99.", [95.2], [99.0]),
    ("Estimated exam score: 80", [], [80.0]),
    ("This student will score about 91 points", [95.2], [91.0]),
    ("Predicted for a student studying 12.5 hours, attendance 88%", [95.2], []),  # inputs are not claims
    ("I can help with exam score predictions.", [], []),
])
def test_find_ungrounded_claims(text, allowed, bad):
    assert guards.find_ungrounded_claims(text, allowed) == bad


# ---- conversation store -----------------------------------------------------------------------

def msg(role, text="x"):
    return {"role": role, "content": text}


def test_store_creates_and_reuses_conversations():
    store = ConversationStore(ttl_seconds=60, max_conversations=5, max_messages=10)
    conv = store.get_or_create(None)
    conv.messages.append(msg("user", "hi"))
    store.save(conv)
    assert store.get_or_create(conv.id).messages == [msg("user", "hi")]
    assert store.delete(conv.id) is True and store.delete(conv.id) is False
    assert store.get_or_create(conv.id).messages == []  # fresh after reset


def test_store_expires_old_conversations(monkeypatch):
    import app.services.sessions as sessions
    clock = {"t": 1000.0}
    monkeypatch.setattr(sessions.time, "monotonic", lambda: clock["t"])
    store = ConversationStore(ttl_seconds=60, max_conversations=5, max_messages=10)
    conv = store.get_or_create("abcdefgh")
    conv.messages.append(msg("user"))
    store.save(conv)
    clock["t"] += 61
    assert store.get_or_create("abcdefgh").messages == []


def test_store_caps_the_number_of_conversations():
    store = ConversationStore(ttl_seconds=600, max_conversations=2, max_messages=10)
    for name in ("conv-aaaa", "conv-bbbb", "conv-cccc"):
        c = store.get_or_create(name)
        c.messages.append(msg("user"))
        store.save(c)
    assert len(store) == 2 and store.get_or_create("conv-aaaa").messages == []  # oldest was evicted


def test_trim_never_starts_with_an_orphan_tool_message():
    store = ConversationStore(ttl_seconds=600, max_conversations=2, max_messages=4)
    conv = store.get_or_create("conv-trim1")
    conv.messages = [msg("user", "1"), {"role": "assistant", "content": None, "tool_calls": []}, msg("tool"), msg("assistant"), msg("user", "2"), msg("assistant")]
    store.save(conv)
    assert conv.messages[0]["role"] == "user" and conv.messages[0]["content"] == "2"


# ---- logging never leaks secrets ---------------------------------------------------------------

def test_log_lines_are_json_and_secrets_are_masked():
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = configure_logging("INFO")
    logger.addHandler(handler)
    try:
        log_event("something", api_key="gsk_SUPERSECRET", Authorization="Bearer gsk_SUPERSECRET", n=3)
    finally:
        logger.removeHandler(handler)
    line = stream.getvalue().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["service"] == "orchestrator" and record["n"] == 3 and "trace_id" in record
    assert "SUPERSECRET" not in line and record["api_key"] == "***"
