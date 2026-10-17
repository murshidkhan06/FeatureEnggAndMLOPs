"""Tests for the shared building blocks (drift, decision ladder, promotion)."""
import numpy as np
import pandas as pd

from src.common.drift import ks_drift_report
from src.common.promotion import compare
from src.common.retrain_decision import decide


def _frames(shift=0.0, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return (pd.DataFrame({"x": rng.normal(0, 1, n)}), pd.DataFrame({"x": rng.normal(shift, 1, n)}))


def test_ks_no_drift_when_same_distribution():
    ref, prod = _frames(0.0)
    rep = ks_drift_report(ref, prod, ["x"])
    assert rep.status.iloc[0] in ("no_evidence_of_drift", "significant_but_small")


def test_ks_detects_big_shift():
    ref, prod = _frames(1.0)
    rep = ks_drift_report(ref, prod, ["x"])
    assert rep.status.iloc[0] == "drift" and rep.p_value.iloc[0] < 0.05


def test_small_sample_gives_no_decision():
    ref, prod = _frames(1.0, n=2000)
    rep = ks_drift_report(ref, prod.head(20), ["x"])
    assert rep.status.iloc[0] == "insufficient_data"


def test_drift_alone_never_means_retrain():
    ref, prod = _frames(1.0)
    rep = ks_drift_report(ref, prod, ["x"])
    d = decide(rep, labels_available=False)
    assert d["decision"] == "INVESTIGATE"          # not RETRAIN_CANDIDATE


def test_retrain_only_with_performance_evidence():
    ref, prod = _frames(1.0)
    rep = ks_drift_report(ref, prod, ["x"])
    assert decide(rep, perf_drop=0.06, labels_available=True)["decision"] == "RETRAIN_CANDIDATE"
    assert decide(rep, perf_drop=0.00, labels_available=True)["decision"] == "MONITOR"


def test_bad_data_quality_blocks_retraining():
    ref, prod = _frames(1.0)
    rep = ks_drift_report(ref, prod, ["x"])
    assert decide(rep, data_quality_ok=False)["decision"] == "FIX_DATA_PIPELINE"


def test_promotion_requires_improvement_and_guardrails():
    champ = {"roc_auc": 0.80, "recall": 0.60}
    assert compare(champ, {"roc_auc": 0.82, "recall": 0.60}, guardrails={"recall": 0.02})["promote"]
    assert not compare(champ, {"roc_auc": 0.802, "recall": 0.60})["promote"]
    assert not compare(champ, {"roc_auc": 0.85, "recall": 0.50}, guardrails={"recall": 0.02})["promote"]
