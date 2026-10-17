"""Tiny helper: every demo script saves its key numbers to reports/facts/<name>.json.
`scripts/build_results.py` later merges them into results.json (machine-readable facts for the course guide)."""
from __future__ import annotations

import json

from .paths import REPORTS


def save_fact(name: str, data: dict) -> None:
    d = REPORTS / "facts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
