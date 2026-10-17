"""Read params.yaml (the single source of truth for numbers such as split dates and hyper-parameters).

Only training / monitoring code uses this file.  The serving image does NOT need PyYAML because the
API never imports this module.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from .paths import ROOT


@lru_cache(maxsize=1)
def load_params() -> dict:
    """Return params.yaml as a nested dict.  Example: load_params()['splits']['train_end'] -> '2021-12-31'."""
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
