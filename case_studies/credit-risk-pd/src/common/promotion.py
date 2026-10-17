"""Champion vs challenger comparison and a tiny local model registry.

Promotion rule (simple and explainable):
  challenger.primary_metric >= champion.primary_metric + min_improvement   (or <= minus it for error metrics)
  AND every guardrail metric is no worse than the champion by more than its tolerance
  AND every absolute limit (e.g. |bias| <= 5%) holds.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib

from .paths import CHALLENGER_PATH, CHAMPION_PATH, REGISTRY_PATH


def compare(champion_metrics: dict, challenger_metrics: dict, primary: str = "roc_auc",
            min_improvement: float = 0.005, guardrails: dict | None = None,
            lower_is_better: tuple | list = (), absolute_limits: dict | None = None) -> dict:
    """guardrails: {metric_name: max_allowed_worsening}  e.g. {'recall': 0.02}

    lower_is_better : metrics where a SMALLER number is better (errors such as WAPE, |bias|).
                      For those, 'improvement' = champion - challenger and 'worsening' = challenger - champion.
    absolute_limits : {metric: max_value} the challenger must satisfy no matter what the champion does,
                      e.g. {'abs_bias_pct': 0.05}  ("never be more than 5% biased")."""
    guardrails = guardrails or {}
    lower = set(lower_is_better)
    sign = -1 if primary in lower else 1
    delta = sign * (challenger_metrics[primary] - champion_metrics[primary])
    checks = {f"{primary}_improvement>={min_improvement}": bool(delta >= min_improvement - 1e-12)}
    for m, tol in guardrails.items():
        worse = (challenger_metrics[m] - champion_metrics[m]) if m in lower else (champion_metrics[m] - challenger_metrics[m])
        checks[f"guardrail_{m}_worsening<={tol}"] = bool(worse <= tol + 1e-12)
    for m, limit in (absolute_limits or {}).items():
        checks[f"limit_{m}<={limit}"] = bool(challenger_metrics[m] <= limit + 1e-12)
    return dict(promote=all(checks.values()), delta=round(delta, 4), checks=checks)


def _read_registry() -> list:
    return json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else []


def register_bootstrap(version: str, metrics: dict) -> None:
    """Record the very first champion in registry.json so that a later rollback has something to roll back TO."""
    REGISTRY_PATH.write_text(json.dumps([dict(version=version, stage="Production", metrics=metrics, approved_by="bootstrap",
                                              reason="first model of the project",
                                              promoted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))], indent=2))


def promote_challenger(approved_by: str, reason: str) -> str:
    """Archive the current champion, copy challenger -> champion, record it in registry.json.
    The new registry entry remembers `previous_version` so that rollback knows exactly WHICH model to restore."""
    ch = joblib.load(CHALLENGER_PATH)
    hist = _read_registry()
    previous = None
    if CHAMPION_PATH.exists():
        old = joblib.load(CHAMPION_PATH)
        previous = old["version"]
        shutil.copy(CHAMPION_PATH, CHAMPION_PATH.with_name(f"model_{previous}.joblib"))
        for h in hist:
            if h.get("stage") == "Production":
                h["stage"] = "Archived"
    shutil.copy(CHALLENGER_PATH, CHAMPION_PATH)
    hist.append(dict(version=ch["version"], stage="Production", metrics=ch.get("metrics", {}), previous_version=previous,
                     approved_by=approved_by, reason=reason,
                     promoted_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    REGISTRY_PATH.write_text(json.dumps(hist, indent=2))
    return ch["version"]


def rollback() -> str | None:
    """Put the model that the CURRENT champion replaced back as champion.  Returns its version (None if impossible).

    Example: registry  v1 (Archived)  ->  v3 (Production, previous_version = v1)   => rollback restores v1 and archives v3.
    (The newest 'Archived' entry is NOT necessarily the right one: after an earlier rollback it can be an older model.)
    The rolled-back model stays on disk as model_<version>.joblib, so a roll-FORWARD is just another copy."""
    hist = _read_registry()
    cur = next((h for h in reversed(hist) if h.get("stage") == "Production"), None)
    if cur is None or not cur.get("previous_version"):
        return None
    prev = cur["previous_version"]
    prev_file = CHAMPION_PATH.with_name(f"model_{prev}.joblib")
    if not prev_file.exists():
        return None
    shutil.copy(CHAMPION_PATH, CHAMPION_PATH.with_name(f"model_{cur['version']}.joblib"))     # keep the demoted model for roll-forward
    shutil.copy(prev_file, CHAMPION_PATH)
    cur["stage"] = "Archived"
    target = next((h for h in reversed(hist) if h["version"] == prev), None)
    if target is None:                                   # the registry lost track of it: re-create a minimal entry
        target = dict(version=prev, metrics={}, approved_by="rollback", reason="restored by rollback", previous_version=None)
        hist.append(target)
    target["stage"] = "Production"
    REGISTRY_PATH.write_text(json.dumps(hist, indent=2))
    return prev
