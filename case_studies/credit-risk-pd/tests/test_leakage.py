"""THE core lesson as executable tests: the safe pipeline never reads the future; the leaky features do."""
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.features.build_features import add_target, assert_no_future_data, build_snapshot_features, filter_available
from src.features.leakage_checks import future_perturbation_test, perturb_future
from src.features.leakage_demo import LEAKY_BUILDERS, add_leaky_columns
from src.governance.feature_lifecycle import FeatureNotApprovedError, check_features_approved
from src.features.definitions import LEAKY_FEATURES, MODEL_FEATURES


def _safe(c, t, l, a):
    return build_snapshot_features(c, t, l, a)


@pytest.mark.parametrize("as_of", ["2022-06-30", "2023-03-31"])
def test_future_perturbation_safe_pipeline_unchanged(raw, as_of):
    """Corrupt every row that is not yet available; the leakage-safe features must be IDENTICAL."""
    r = future_perturbation_test(_safe, raw, as_of)
    assert r["passed"], r["changed_columns"]
    assert r["n_future_tx_rows"] > 50_000                      # the test really corrupted a lot of future data


@pytest.mark.parametrize("name", list(LEAKY_BUILDERS))
def test_future_perturbation_catches_every_leaky_feature(raw, name):
    assert not future_perturbation_test(LEAKY_BUILDERS[name], raw, "2022-06-30")["passed"]


def test_perturbation_really_changes_future_rows(raw):
    tx2, ls2 = perturb_future(raw["transactions"], raw["loan_status"], "2022-06-30")
    fut = raw["transactions"]["txn_date"] > "2022-06-30"
    assert not tx2.loc[fut, "balance_after"].equals(raw["transactions"].loc[fut, "balance_after"])
    assert tx2.loc[~fut].equals(raw["transactions"].loc[~fut])          # the past is untouched


def test_assert_no_future_data_guard():
    df = pd.DataFrame({"txn_date": pd.to_datetime(["2023-03-30", "2023-03-31"])})
    assert_no_future_data(df, "2023-03-31", "txn_date")                  # ok
    with pytest.raises(AssertionError, match="future data detected"):
        assert_no_future_data(df, "2023-03-30", "txn_date")


def test_availability_rule_for_loan_status(raw):
    ls = raw["loan_status"]
    feb = ls[ls["month"] == "2023-02-28"]
    assert len(filter_available(feb, "2023-02-28", "available_at")) == 0          # published 2023-03-05: not usable on 2023-02-28
    assert len(filter_available(feb, "2023-03-31", "available_at")) == len(feb)   # usable a month later


def test_leaky_balance_looks_too_good_and_the_honest_one_does_not(raw):
    as_of = pd.Timestamp("2022-12-31")
    snap = add_target(build_snapshot_features(raw["customers"], raw["transactions"], raw["loan_status"], as_of, raw["defaults"]), raw["defaults"])
    snap["split"] = "x"
    snap = add_leaky_columns(snap, raw)
    auc_leaky = roc_auc_score(snap["default_next_6m"], -snap["average_balance_leaky"])
    auc_ok = roc_auc_score(snap["default_next_6m"], -snap["average_balance"])
    assert auc_leaky > 0.85 and auc_ok < 0.75 and auc_leaky - auc_ok > 0.15
    assert roc_auc_score(snap["default_next_6m"], snap["dpd_target_period"]) > 0.99         # the label hiding in plain sight


def test_registry_gate_blocks_every_leaky_feature():
    for f in LEAKY_FEATURES:
        with pytest.raises(FeatureNotApprovedError):
            check_features_approved([f])
    assert check_features_approved(MODEL_FEATURES) == MODEL_FEATURES
