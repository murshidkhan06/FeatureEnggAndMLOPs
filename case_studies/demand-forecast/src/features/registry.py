"""Feature metadata registry: the contract between data owners and modellers.

configs/feature_registry.yaml describes EVERY feature: definition, source, owner, calculation logic,
whether it is available at prediction time, and whether it is Approved or Rejected (with the reason).

WHY a registry?  Without one, someone eventually adds "temperature" (actual) because it improves
the validation score - and the model silently becomes unusable in production.
Rule enforced by training:  a feature that is not Approved cannot be used.

Tiny example
------------
>>> check_features_approved(["lag_1", "rolling_mean_7"])        # fine, returns None
>>> check_features_approved(["lag_1", "temperature"])           # raises FeatureNotApproved
"""
from __future__ import annotations

import pandas as pd
import yaml

from src.common.paths import CONFIGS

REGISTRY_FILE = CONFIGS / "feature_registry.yaml"


class FeatureNotApproved(ValueError):
    """Raised when a model tries to use a feature that is Rejected / unknown / not an input feature."""


def load_registry(path=REGISTRY_FILE) -> pd.DataFrame:
    data = yaml.safe_load(open(path, encoding="utf-8"))
    return pd.DataFrame(data["features"]).set_index("name", drop=False)


def check_features_approved(features: list[str], registry: pd.DataFrame | None = None) -> None:
    """Raise FeatureNotApproved (with the reason) if any feature is not an Approved model input."""
    reg = load_registry() if registry is None else registry
    problems = []
    for f in features:
        if f not in reg.index:
            problems.append(f"{f}: not in the registry (register it with an owner and an availability statement first)")
        elif reg.loc[f, "role"] != "feature":
            problems.append(f"{f}: role is '{reg.loc[f, 'role']}', not a model input feature")
        elif reg.loc[f, "status"] != "Approved":
            problems.append(f"{f}: {reg.loc[f, 'status']} - {reg.loc[f, 'status_reason']}")
    if problems:
        raise FeatureNotApproved("Feature gate failed:\n  - " + "\n  - ".join(problems))


def approved_features(registry: pd.DataFrame | None = None) -> list[str]:
    reg = load_registry() if registry is None else registry
    return reg[(reg.role == "feature") & (reg.status == "Approved")]["name"].tolist()


if __name__ == "__main__":
    reg = load_registry()
    print(reg[["role", "status", "prediction_time_availability"]].to_string(max_colwidth=60))
