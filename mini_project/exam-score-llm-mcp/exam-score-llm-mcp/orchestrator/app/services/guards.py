"""app/services/guards.py

Code-level safety checks that do NOT depend on the LLM behaving well.

  1. parse_arguments        -> is the LLM's tool-call JSON even valid?
  2. validate_arguments     -> required fields present? right types / ranges? no unknown keys?
  3. check_provenance       -> did the USER actually write this value? (catches invented defaults)
  4. find_ungrounded_claims -> does the reply state a "predicted score" the model never produced?

None of this duplicates the ML pipeline's preprocessing: we only check that the RAW values
are present and plausible. Scaling, encoding, imputation, PCA etc. stay inside the fitted
scikit-learn Pipeline behind the ML API.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator


@dataclass
class ArgIssue:
    field: str
    kind: str  # "missing" | "unknown" | "invalid" | "ungrounded"
    message: str


# ---- 1. parsing --------------------------------------------------------------------------

def parse_arguments(raw: str) -> dict | None:
    """Return the arguments as a dict, or None if the LLM produced invalid JSON / not an object."""
    try:
        value = json.loads(raw) if raw and raw.strip() else {}
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


_NUMERIC_STRING = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")


def coerce_numeric_strings(schema: dict, args: dict) -> dict:
    """LLMs sometimes send "85" instead of 85. Convert ONLY clean numeric strings for fields
    the schema declares as numbers. Anything else is left alone and fails validation."""
    props = schema.get("properties", {})
    out = dict(args)
    for key, value in args.items():
        ptype = props.get(key, {}).get("type")
        if isinstance(value, str) and ptype in ("number", "integer") and _NUMERIC_STRING.match(value):
            number = float(value)
            out[key] = int(number) if ptype == "integer" and number.is_integer() else number
    return out


# ---- 2. schema validation ----------------------------------------------------------------

def validate_arguments(schema: dict, args: dict) -> list[ArgIssue]:
    """Validate against the JSON-Schema that the MCP server itself published for the tool."""
    props = schema.get("properties", {})
    issues: list[ArgIssue] = []

    for name in schema.get("required", []):
        if name not in args or args[name] is None or (isinstance(args[name], str) and not args[name].strip()):
            issues.append(ArgIssue(name, "missing", "no value was provided"))
    for name in args:
        if name not in props:
            issues.append(ArgIssue(name, "unknown", "this is not a field the model accepts"))

    present = {k: v for k, v in args.items() if k in props and v is not None}
    validator = Draft202012Validator(schema)
    for error in validator.iter_errors(present):
        if error.validator == "required":
            continue  # handled above
        field = str(error.path[0]) if error.path else "(input)"
        issues.append(ArgIssue(field, "invalid", _friendly(error.message)))
    return issues


def _friendly(message: str) -> str:
    return message if len(message) < 140 else message[:137] + "..."


# ---- 3. provenance -----------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


# A number can only describe a field if the user actually talked about that topic. This catches the
# coincidence problem: "completed 8 assignments" must not be accepted as "shoe size = 8".
# Deliberately crude (keywords, not understanding) - it is a second net, not a replacement for good prompts.
FIELD_CUES: dict[str, re.Pattern] = {
    "study_hours": re.compile(r"stud|hour|hrs", re.I),
    "attendance_pct": re.compile(r"attend|present|class", re.I),
    "mock_test_1": re.compile(r"mock|test|mark|score", re.I),
    "mock_test_2": re.compile(r"mock|test|mark|score", re.I),
    "mock_test_3": re.compile(r"mock|test|mark|score", re.I),
    "shoe_size": re.compile(r"shoe", re.I),
    "lucky_number": re.compile(r"lucky", re.I),
    "enrollment_date": re.compile(r"enrol|admi|join|start|since|date|regist", re.I),
}
NOT_FOUND_MESSAGE = "I could not find this in what you wrote, so I will not guess it"


def check_provenance(args: dict, user_text: str) -> list[ArgIssue]:
    """Every value the LLM passes must be traceable to something the user typed.

    This is a deliberately simple, conservative check: numbers must appear as digits in the
    user's messages; text values must appear (case-insensitive) as words; dates need their
    year and day present. A false alarm just means the bot asks the user to restate a value
    - a SAFE failure. A missed invention is the unsafe failure we are guarding against.
    """
    numbers = {float(n) for n in _NUMBER.findall(user_text)}
    lowered = user_text.lower()
    issues: list[ArgIssue] = []

    for key, value in args.items():
        cue = FIELD_CUES.get(key)
        if cue is not None and not cue.search(user_text):
            issues.append(ArgIssue(key, "ungrounded", NOT_FOUND_MESSAGE))
            continue
        if isinstance(value, bool):
            ok = False
        elif isinstance(value, (int, float)):
            ok = any(abs(value - n) < 1e-9 for n in numbers)
        elif isinstance(value, str):
            date_match = _ISO_DATE.match(value.strip())
            if date_match:
                year, _month, day = (int(g) for g in date_match.groups())
                ok = float(year) in numbers and float(day) in numbers
            else:
                ok = re.search(rf"\b{re.escape(value.strip().lower())}\b", lowered) is not None
        else:
            ok = False
        if not ok:
            issues.append(ArgIssue(key, "ungrounded", NOT_FOUND_MESSAGE))  # never echo the LLM's invented value
    return issues


# ---- 3b. unit check (a small DOMAIN guard, as a teaching example) ---------------------------

_PER_DAY = re.compile(r"\b(daily|per\s+day|a\s+day|each\s+day|every\s+day|/\s*day)\b", re.IGNORECASE)
_PER_WEEK = re.compile(r"\b(weekly|per\s+week|a\s+week|each\s+week|every\s+week|/\s*week)\b", re.IGNORECASE)


def check_unit_mismatch(schema: dict, args: dict, latest_message: str) -> list[ArgIssue]:
    """The model's `study_hours` is hours per WEEK. If the user's latest message talks about
    hours per DAY (and not per week), refuse to guess the conversion and ask instead.

    Why not just multiply by 7? Because the LLM/orchestrator must never silently invent a
    value: 6 hours daily is 42 per week, but the model only accepts up to 40, and the user may
    not study on weekends. Only the user knows. (Lab exercise: generalise this guard.)
    """
    issues: list[ArgIssue] = []
    if _PER_DAY.search(latest_message) and not _PER_WEEK.search(latest_message):
        for name, spec in schema.get("properties", {}).items():
            if name in args and "hours" in name and "WEEK" in str(spec.get("description", "")):
                issues.append(ArgIssue(
                    name, "unit",
                    "you described hours per DAY, but this field is hours per WEEK - please tell me the weekly total",
                ))
    return issues


# ---- 4. output guard ---------------------------------------------------------------------

# Matches phrases like "predicted final score is 95.2", "estimated exam score: 80", "will score 91".
_CLAIM = re.compile(
    r"(?:(?:predicted|estimated|expected|forecast(?:ed)?)(?:\s+(?:final|exam))*\s+score"
    r"(?:\s+(?:is|was|of|would\s+be|to\s+be))?\s*[:=]?\s*"
    r"|(?:will|would|likely\s+to)\s+score\s+(?:about|around|approximately)?\s*)"
    r"(\d{1,3}(?:\.\d+)?)",
    re.IGNORECASE,
)


def find_ungrounded_claims(text: str, allowed_scores: list[float]) -> list[float]:
    """Return any 'predicted score' numbers in `text` that are not among `allowed_scores`
    (the values the ML model really returned)."""
    bad: list[float] = []
    for match in _CLAIM.finditer(text or ""):
        written = match.group(1)
        number = float(written)
        # "95" is a fair rounding of 95.2, "99" is not. Whole numbers get a +-0.5 tolerance.
        tolerance = 0.51 if "." not in written else 0.051
        if not any(abs(number - allowed) < tolerance for allowed in allowed_scores):
            bad.append(number)
    return bad


def display_value(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)
