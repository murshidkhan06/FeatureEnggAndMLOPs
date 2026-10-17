"""Data-quality gate, KS-based drift table, credit decision policy, label-lag helpers."""
import numpy as np
import pandas as pd
import pytest

from src.common.drift import ks_drift_report
from src.features.definitions import BASE_FEATURES
from src.monitoring.credit_policy import decide_credit_risk
from src.monitoring.data_quality import contract_ranges, validate_batch
from src.monitoring.proxy_metrics import early_delinquency_rate, label_available_on, labels_mature, score_shift

from .conftest import needs_production

KEY = ["salary_credit_frequency", "credit_utilization", "income_stability", "missed_payment_count", "balance_trend"]


def _report(drifting=(), small=()):
    rows = [dict(feature=f, status="drift" if f in drifting else "significant_but_small" if f in small else "no_evidence_of_drift") for f in KEY + ["account_age"]]
    return pd.DataFrame(rows)


def test_policy_needs_breadth_and_outcome_evidence():
    four = KEY[:4]
    assert decide_credit_risk(_report(four), KEY, True, pseudo={"rate_ratio": 2.1})["decision"] == "RETRAIN_CANDIDATE"
    assert decide_credit_risk(_report(four), KEY, True)["decision"] == "INVESTIGATE"                                  # drift alone never retrains
    lab = dict(auc_drop=0.09, calibration_ratio=1.1, contact_rate_ratio=1.0, recall_drop=0.0)
    assert decide_credit_risk(_report(four), KEY, True, labelled=lab)["decision"] == "RETRAIN_CANDIDATE"
    assert decide_credit_risk(_report(KEY[:1]), KEY, True, labelled=lab)["decision"] == "INVESTIGATE"                  # not broad: look elsewhere
    stable = dict(auc_drop=0.0, calibration_ratio=1.0, contact_rate_ratio=1.0, recall_drop=0.0)
    assert decide_credit_risk(_report(four), KEY, True, labelled=stable)["decision"] == "INVESTIGATE"                  # inputs moved, model still fine


def test_policy_data_quality_comes_first_and_small_shifts_do_nothing():
    d = decide_credit_risk(_report(KEY), KEY, False, ["salary_credit_frequency: 40% nulls"], pseudo={"rate_ratio": 3.0})
    assert d["decision"] == "FIX_DATA_PIPELINE"
    assert decide_credit_risk(_report(small=["account_age"]), KEY, True)["decision"] == "MONITOR"
    assert decide_credit_risk(_report(), KEY, True)["decision"] == "NO_ACTION"


def test_policy_scheduled_retrain():
    assert decide_credit_risk(_report(), KEY, True, scheduled_due=True)["decision"] == "RETRAIN_CANDIDATE"
    assert decide_credit_risk(_report(), KEY, False, scheduled_due=True)["decision"] == "FIX_DATA_PIPELINE"        # even a schedule waits for clean data


def test_significant_but_small_drift_is_labelled_correctly():
    rng = np.random.default_rng(0)
    ref, prod = pd.DataFrame({"x": rng.normal(0, 1, 20000)}), pd.DataFrame({"x": rng.normal(0.05, 1, 20000)})
    r = ks_drift_report(ref, prod, ["x"]).iloc[0]
    assert r.p_value < 0.05 and r.ks_statistic < 0.10 and r.status == "significant_but_small"


def test_label_lag_arithmetic():
    assert label_available_on("2024-01-31") == pd.Timestamp("2024-07-31")
    assert labels_mature("2024-01-31", "2024-07-31") and not labels_mature("2024-01-31", "2024-07-30")


def test_early_delinquency_pseudo_label(raw):
    ls = raw["loan_status"]
    ids = ls["customer_id"].unique().tolist()
    assert early_delinquency_rate(ls, ids, "2024-01-31", "2024-03-31") is None                    # 3 months not published yet
    r = early_delinquency_rate(ls, ids, "2024-01-31", "2024-06-30")
    assert 0.0 < r < 0.5


def test_score_shift_contact_rate():
    art = {"threshold": 0.5}
    s = score_shift(art, np.array([0.1, 0.2, 0.6, 0.9]), ref_mean=0.2, top_pct=0.10)
    assert s["contact_rate"] == 0.5 and s["contact_rate_ratio"] == 5.0 and s["mean_pd_ratio"] == pytest.approx(2.25, abs=0.01)


@needs_production
def test_data_quality_gate_on_the_three_production_batches():
    from src.common.paths import DATA_PRODUCTION, REFERENCE_PATH
    ref = pd.read_csv(REFERENCE_PATH)
    ok = validate_batch(pd.read_csv(DATA_PRODUCTION / "batch_normal.csv"), ref)
    assert ok["ok"], ok["problems"]
    assert validate_batch(pd.read_csv(DATA_PRODUCTION / "batch_drifted.csv"), ref)["ok"]               # drift is NOT a data-quality failure
    bad = validate_batch(pd.read_csv(DATA_PRODUCTION / "batch_dq_issue.csv"), ref)
    assert not bad["ok"] and any("salary_credit_frequency" in p for p in bad["problems"])


def test_data_quality_detects_range_duplicates_and_mixed_dates():
    ref = pd.DataFrame({f: [0.5] * 5 for f in BASE_FEATURES})
    b = pd.DataFrame({f: [0.5, 0.5] for f in BASE_FEATURES})
    b["customer_id"], b["features_as_of"] = [1, 1], ["2023-03-31", "2023-04-30"]
    b.loc[0, "loan_payment_ratio"] = 4.0
    r = validate_batch(b, ref)
    txt = " ".join(r["problems"])
    assert not r["ok"] and "duplicated" in txt and "different features_as_of" in txt and "outside the allowed range" in txt
    assert contract_ranges()["loan_payment_ratio"] == (0, 1)
    assert not validate_batch(b.drop(columns=["balance_trend"]), ref)["ok"]
