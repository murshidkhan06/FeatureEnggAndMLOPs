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


def _fake_registry(tmp_path, monkeypatch):
    """Point promotion.py at temporary files and install a first champion v1 exactly like `train --bootstrap-champion` does."""
    import joblib
    from src.common import promotion
    champ, chall, reg = tmp_path / "champion.joblib", tmp_path / "challenger.joblib", tmp_path / "registry.json"
    for name, path in [("CHAMPION_PATH", champ), ("CHALLENGER_PATH", chall), ("REGISTRY_PATH", reg)]:
        monkeypatch.setattr(promotion, name, path)
    joblib.dump({"version": "v1", "metrics": {"roc_auc": 0.76}}, champ)
    promotion.register_bootstrap("v1", {"roc_auc": 0.76})
    return promotion, champ, chall, reg


def test_promote_then_rollback_restores_the_bootstrap_champion(tmp_path, monkeypatch):
    import joblib
    promotion, champ, chall, _ = _fake_registry(tmp_path, monkeypatch)
    joblib.dump({"version": "v2", "metrics": {"roc_auc": 0.77}}, chall)
    assert promotion.promote_challenger("Test Approver", "unit test") == "v2"
    assert joblib.load(champ)["version"] == "v2"
    assert promotion.rollback() == "v1"
    assert joblib.load(champ)["version"] == "v1"


def test_two_promote_rollback_cycles_restore_the_right_model(tmp_path, monkeypatch):
    """Regression test: in the 2nd cycle the newest 'Archived' entry is v2, but rollback must restore v1 (the model v3 REPLACED)."""
    import joblib, json
    promotion, champ, chall, reg = _fake_registry(tmp_path, monkeypatch)
    for new in ("v2", "v3"):
        joblib.dump({"version": new, "metrics": {"roc_auc": 0.78}}, chall)
        assert promotion.promote_challenger("Test Approver", "cycle") == new
        assert joblib.load(champ)["version"] == new
        assert promotion.rollback() == "v1"
        assert joblib.load(champ)["version"] == "v1"
    stages = {h["version"]: h["stage"] for h in json.loads(reg.read_text())}
    assert stages["v1"] == "Production" and stages["v2"] == stages["v3"] == "Archived"


def test_rollback_without_history_returns_none(tmp_path, monkeypatch):
    promotion, *_ = _fake_registry(tmp_path, monkeypatch)
    assert promotion.rollback() is None


def test_compare_supports_lower_is_better_and_absolute_limits():
    from src.common.promotion import compare
    assert compare({"err": 0.20}, {"err": 0.10}, primary="err", lower_is_better=["err"])["promote"]
    assert not compare({"err": 0.20}, {"err": 0.10}, primary="err", lower_is_better=["err"], absolute_limits={"err": 0.05})["promote"]
