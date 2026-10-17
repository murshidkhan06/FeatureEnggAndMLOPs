"""Read configs/params.yaml and configs/retrain_policy.yaml (one place, so scripts never hard-code numbers)."""
from __future__ import annotations

import yaml

from .paths import CONFIGS


def load_params() -> dict:
    return yaml.safe_load(open(CONFIGS / "params.yaml", encoding="utf-8"))


def load_policy() -> dict:
    return yaml.safe_load(open(CONFIGS / "retrain_policy.yaml", encoding="utf-8"))
