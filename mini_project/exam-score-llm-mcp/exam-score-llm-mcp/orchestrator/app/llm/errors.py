"""app/llm/errors.py -- one error type for every way an LLM provider can fail.

Each error has a stable `code` (for the React UI and for tests), a safe human `message`,
and the HTTP status the orchestrator should answer with. Messages never contain secrets.
"""
from __future__ import annotations


class LLMError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
