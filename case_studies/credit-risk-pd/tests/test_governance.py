"""Feature registry + lifecycle gate (src/governance/feature_lifecycle.py)."""
import copy

import pytest

from src.features.definitions import MODEL_FEATURES
from src.governance.feature_lifecycle import (FeatureNotApprovedError, LifecycleError, advance, check_features_approved, latest_entries,
                                              load_registry, validation_failures)

REG = load_registry()
RULES = REG["validation_rules"]
REQUIRED = ["name", "definition", "source", "data_type", "owner", "calculation_logic", "prediction_time_availability", "status", "version", "stage",
            "availability", "history"]
GOOD = {"leakage_test": "pass", "null_rate": 0.0, "availability_rule_declared": True, "owner_assigned": True}


def _fresh(name="salary_credit_frequency", stage="Development"):
    f = copy.deepcopy(latest_entries(REG)[name])
    f["stage"], f["status"], f["history"] = stage, "Development", [h for h in f["history"] if h["stage"] == "Development"]
    return f


def test_every_feature_has_complete_metadata():
    for f in REG["features"]:
        assert all(k in f and f[k] not in (None, "") for k in REQUIRED), f["name"]
        assert f["status"] in ("Approved", "Rejected") and f["history"]


def test_every_model_feature_is_approved_in_production():
    assert check_features_approved(MODEL_FEATURES, REG)
    for n in MODEL_FEATURES:
        e = latest_entries(REG)[n]
        assert e["status"] == "Approved" and e["stage"] == "Production" and e["availability"]["rule"].startswith("available_at")


def test_rejected_leaky_feature_and_its_approved_replacement():
    leaky = [f for f in REG["features"] if f["name"] == "average_balance_leaky"][0]
    assert leaky["status"] == "Rejected"
    assert "post-default" in leaky["rejection_reason"].lower() or "future" in leaky["rejection_reason"].lower()
    v = {f["version"]: f for f in REG["features"] if f["name"] == "average_balance"}
    assert v[2]["status"] == "Approved" and v[2].get("replaces", "").startswith("average_balance_leaky")


def test_training_refuses_unapproved_or_unregistered_features():
    with pytest.raises(FeatureNotApprovedError, match="Rejected"):
        check_features_approved(["average_balance", "average_balance_leaky"], REG)
    with pytest.raises(FeatureNotApprovedError, match="not in the feature registry"):
        check_features_approved(["shoe_size"], REG)
    with pytest.raises(FeatureNotApprovedError):
        check_features_approved(["segment"], REG)                                     # sensitive attribute is Rejected


def test_fit_model_calls_the_gate():
    import pandas as pd
    from src.models.train import fit_model
    with pytest.raises(FeatureNotApprovedError):
        fit_model(pd.DataFrame({"average_balance_leaky": [1.0], "default_next_6m": [0]}), ["average_balance_leaky"], "logistic")


def test_legal_path_to_production():
    f = _fresh()
    for stage in ("Validation", "Approval", "Production"):
        f = advance(f, stage, "tester", "2022-11-01", RULES, GOOD if stage == "Validation" else None)
    assert f["stage"] == "Production" and f["status"] == "Approved" and [h["stage"] for h in f["history"]][-1] == "Production"


def test_cannot_skip_validation():
    with pytest.raises(LifecycleError, match="cannot go Development -> Production"):
        advance(_fresh(), "Production", "tester", "2022-11-01", RULES)


@pytest.mark.parametrize("bad,why", [({"leakage_test": "FAIL"}, "leakage_test"), ({"null_rate": 0.35}, "null_rate"), ({"null_rate": None}, "null_rate")])
def test_gate_blocks_failed_checks(bad, why):
    f = advance(_fresh(), "Validation", "tester", "2022-11-01", RULES, {**GOOD, **bad})
    with pytest.raises(LifecycleError, match=why):
        advance(f, "Approval", "committee", "2022-11-08", RULES)


def test_gate_blocks_missing_owner_and_missing_availability_rule():
    f = advance(_fresh(), "Validation", "tester", "2022-11-01", RULES, GOOD)
    no_owner = {**f, "owner": ""}
    assert "no owner assigned" in validation_failures(no_owner, RULES)
    no_rule = {**f, "availability": {"lag_days": None, "rule": ""}}
    assert "availability rule not declared" in validation_failures(no_rule, RULES)


def test_rejected_is_a_dead_end_and_advance_does_not_mutate_input():
    f = _fresh()
    v = advance(f, "Validation", "t", "2022-11-01", RULES, GOOD)
    assert f["stage"] == "Development"                                                # original untouched
    r = advance(v, "Rejected", "committee", "2022-11-08", RULES)
    with pytest.raises(LifecycleError):
        advance(r, "Production", "x", "2022-11-09", RULES)
