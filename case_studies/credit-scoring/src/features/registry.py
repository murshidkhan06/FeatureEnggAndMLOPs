"""Feature registry (feature metadata) helpers + the approval check that guards training.

WHY a registry?  In a regulated business every model input needs an owner, a definition, proof that it is
available when the decision is made, and a recorded approval.  A model that silently uses a leaky or sensitive
column can be illegal (sensitive attribute) or useless (leakage) - so training REFUSES any feature that is
not 'Approved' in configs/feature_registry.yaml.

Tiny example
    check_features_approved(["log_income", "gender"])  ->  raises FeatureNotApproved (gender is Rejected: sensitive)
"""
from __future__ import annotations

import yaml

from src.common.paths import CONFIGS


class FeatureNotApproved(ValueError):
    """Raised when a model asks for a feature the registry has not approved."""


def load_registry(path=None) -> dict[str, dict]:
    path = path or CONFIGS / "feature_registry.yaml"
    items = yaml.safe_load(open(path, encoding="utf-8"))["features"]
    return {r["name"]: r for r in items}


def check_features_approved(features: list[str], registry: dict[str, dict] | None = None) -> None:
    """Raise FeatureNotApproved with a readable reason for every feature that is unknown or not Approved."""
    registry = registry or load_registry()
    problems = []
    for f in features:
        rec = registry.get(f)
        if rec is None:
            problems.append(f"{f}: not in the feature registry (no owner / definition / approval)")
        elif rec["status"] != "Approved":
            problems.append(f"{f}: {rec['status']} [{rec['decision_category']}] - {rec['reason']}")
        elif rec.get("sensitivity") in ("sensitive_attribute", "restricted_attribute"):
            problems.append(f"{f}: sensitivity={rec['sensitivity']} may not be a model input")
    if problems:
        raise FeatureNotApproved("Feature approval check FAILED:\n  - " + "\n  - ".join(problems))


def approved_features(registry: dict[str, dict] | None = None) -> list[str]:
    registry = registry or load_registry()
    return [n for n, r in registry.items() if r["status"] == "Approved"]


def load_selected(path=None) -> list[str]:
    path = path or CONFIGS / "selected_features.yaml"
    return list(yaml.safe_load(open(path, encoding="utf-8"))["selected_features"])
