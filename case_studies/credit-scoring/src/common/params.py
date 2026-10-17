"""Read configs/params.yaml (one place for every tunable number: seeds, split dates, thresholds).

WHY a params file?  So that DVC can see when a parameter changed and re-run only the stages that depend
on it, and so a reviewer can read all assumptions in one page.  (Needs PyYAML - a development
dependency; the serving image does not import this module.)
"""
from __future__ import annotations

import yaml

from .paths import CONFIGS


def load_params() -> dict:
    return yaml.safe_load((CONFIGS / "params.yaml").read_text(encoding="utf-8"))
