"""Feature metadata + the feature gate: training may only use Approved features."""
import pytest

from src.features.definitions import ALL_CANDIDATE_FEATURES, LEAKY_FEATURES
from src.features.registry import FeatureNotApproved, approved_features, check_features_approved, load_registry
from src.features.select_features import load_selected

REQUIRED_FIELDS = ["name", "definition", "source", "data_type", "owner", "calculation_logic",
                   "prediction_time_availability", "status", "status_reason", "version"]


def test_every_feature_is_documented():
    reg = load_registry()
    for f in ALL_CANDIDATE_FEATURES + LEAKY_FEATURES:
        assert f in reg.index, f"{f} missing in feature_registry.yaml"
    for field in REQUIRED_FIELDS:
        assert reg[field].notna().all() and (reg[field].astype(str).str.len() > 0).all(), field


def test_leaky_features_are_rejected_with_a_reason():
    reg = load_registry()
    for f in ("temperature", "rolling_mean_7_centered", "rolling_mean_7_noshift", "sales_today"):
        assert reg.loc[f, "status"] == "Rejected" and len(reg.loc[f, "status_reason"]) > 10


def test_gate_accepts_selected_features():
    check_features_approved(load_selected())


@pytest.mark.parametrize("bad", ["temperature", "rolling_mean_7_centered", "not_a_feature"])
def test_gate_blocks_rejected_or_unknown(bad):
    with pytest.raises(FeatureNotApproved):
        check_features_approved(["lag_1", bad])


def test_target_is_not_a_model_input():
    with pytest.raises(FeatureNotApproved):
        check_features_approved(["sales"])


def test_approved_list_is_non_trivial():
    a = approved_features()
    assert "lag_1" in a and "temperature" not in a and len(a) >= 15
