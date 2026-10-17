"""Monitoring: data-quality rules, the decision ladder on real batches, performance monitor."""
import pandas as pd

import pytest
from tests.conftest import needs_data, needs_model
from src import project as P
from src.common.drift import ks_drift_report
from src.common.retrain_decision import decide
from src.features.build_features import engineer
from src.monitoring.data_quality import validate_batch

pytestmark = [needs_data]


def _ref():
    from src.common.paths import REFERENCE_PATH
    return pd.read_csv(REFERENCE_PATH)


def test_reference_has_all_drift_features():
    assert set(P.DRIFT_FEATURES) <= set(_ref().columns) and set(P.DRIFT_FEATURES) <= set(P.INPUT_COLUMNS)


def test_normal_batch_passes_data_quality(batches):
    assert validate_batch(batches[0])["ok"] and validate_batch(batches[1])["ok"]


def test_unit_change_is_caught_by_data_quality(batches):
    bad = batches[0].assign(monthly_income=batches[0].monthly_income * 12)
    r = validate_batch(bad)
    assert not r["ok"] and r["checks"]["unit_sanity"] == "FAIL" and r["income_unit_ratio"] > 10


def test_renamed_column_and_null_explosion_are_caught(batches):
    renamed = batches[0].rename(columns={"monthly_debt": "monthly_debt_inr"})
    assert validate_batch(renamed)["checks"]["schema"] == "FAIL"
    nulls = batches[0].copy()
    nulls.loc[nulls.sample(frac=0.5, random_state=0).index, "monthly_debt"] = None
    assert not validate_batch(nulls)["ok"]
    cat = batches[0].assign(employment_type="Salaried")
    assert validate_batch(cat)["checks"]["categories"] == "FAIL"


def test_normal_batch_has_a_significant_but_small_shift_and_no_action(batches):
    rep = ks_drift_report(_ref(), engineer(batches[0]), P.DRIFT_FEATURES)
    st = rep.set_index("feature").status
    assert st["credit_history_years"] == "significant_but_small" if "credit_history_years" in st else True
    assert "drift" not in set(st)
    assert decide(rep)["decision"] == "NO_ACTION"


def test_drifted_batch_shows_drift_but_without_labels_the_ladder_says_investigate(batches):
    rep = ks_drift_report(_ref(), engineer(batches[1]), P.DRIFT_FEATURES)
    assert (rep.status == "drift").sum() >= 3
    assert decide(rep, labels_available=False)["decision"] == "INVESTIGATE"


@needs_model
def test_labelled_performance_drop_turns_investigate_into_retrain_candidate(batches, champion):
    from src.monitoring.perf_monitor import labelled_performance
    if champion["version"] != "v1":
        pytest.skip("champion was replaced by retraining (its reference AUC is a hold-out AUC); this test is written for the bootstrap champion v1")
    normal = labelled_performance(batches[0], champion)
    drifted = labelled_performance(batches[1], champion)
    assert normal["perf_drop"] < 0.03 <= drifted["perf_drop"]
    assert drifted["calibration_gap"] < -0.03                                 # the model now UNDER-predicts defaults
    rep = ks_drift_report(_ref(), engineer(batches[1]), P.DRIFT_FEATURES)
    assert decide(rep, perf_drop=drifted["perf_drop"], labels_available=True)["decision"] == "RETRAIN_CANDIDATE"
    rep_n = ks_drift_report(_ref(), engineer(batches[0]), P.DRIFT_FEATURES)
    assert decide(rep_n, perf_drop=normal["perf_drop"], labels_available=True)["decision"] in ("NO_ACTION", "MONITOR")


def test_unit_change_batch_gets_fix_data_pipeline_not_retrain(batches):
    bad = batches[0].assign(monthly_income=batches[0].monthly_income * 12)
    rep = ks_drift_report(_ref(), engineer(bad), P.DRIFT_FEATURES)
    assert decide(rep, data_quality_ok=validate_batch(bad)["ok"], perf_drop=0.10, labels_available=True)["decision"] == "FIX_DATA_PIPELINE"
