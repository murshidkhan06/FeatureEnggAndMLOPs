"""Model tests (need the trained champion and data/processed)."""
import numpy as np
import pytest

from tests.conftest import needs_data, needs_model
from src import project as P
from src.common.params import load_params
from src.features.definitions import EXPECTED_SIGN
from src.features.registry import check_features_approved
from src.models.metrics import business_view, calibration_table, credit_metrics, ks_statistic, recall_at_approval_rate
from src.models.pipeline import encoded_to_feature
from src.models.reason_codes import contribution_matrix, reason_codes, standardized_coefficients
import json
from sklearn.metrics import roc_auc_score
from src.common.paths import CONFIGS

pytestmark = [needs_data, needs_model]


def test_metric_hand_examples():
    # AUC: good=[0.1,0.2], bad=[0.15,0.9]: 3 of the 4 (bad, good) pairs are ranked correctly -> 0.75
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.15, 0.9])
    assert roc_auc_score(y, p) == 0.75 and credit_metrics(y, p)["gini"] == 0.5
    # KS: bad scores [0.15, 0.9] vs good [0.1, 0.2] -> largest CDF gap is 0.5
    assert ks_statistic(y, p) == pytest.approx(0.5)
    # recall at 50% approval: decline the highest 50% of scores (0.9 and 0.2?)  cutoff = median(0.175) -> declines 0.2 and 0.9 -> 1 of 2 defaulters
    assert recall_at_approval_rate(y, p, 0.5) == 0.5


def test_business_view_hand_example():
    # 5 applicants, decline the riskiest 40%: two declined; one defaulter (loan 100k) and one good customer (loan 50k)
    la, y, p = [100_000, 50_000, 10_000, 10_000, 10_000], [1, 0, 0, 0, 1], [0.9, 0.8, 0.1, 0.1, 0.1]
    b = business_view(la, y, p, approval_rate=0.6, lgd=0.6, margin=0.08)
    assert b["loss_avoided"] == 60_000 and b["margin_lost"] == 4_000 and b["net_benefit"] == 56_000


def test_champion_meets_quality_gate(champion):
    gate = json.loads((CONFIGS / "quality_gate.json").read_text())["min"]
    for k, v in gate.items():
        assert champion["metrics"][k] >= v, k
    assert champion["metrics"]["gini"] == pytest.approx(2 * champion["metrics"]["roc_auc"] - 1, abs=1e-3)


def test_champion_uses_only_approved_nonsensitive_features(champion):
    check_features_approved(champion["input_columns"])
    assert not {"gender", "marital_status", "age", "location", "interest_rate_offered"} & set(champion["input_columns"])


def test_honest_auc_is_in_a_believable_range(champion):
    assert 0.70 <= champion["metrics"]["roc_auc"] <= 0.85            # > 0.9 in lending would be a leakage alarm


def test_model_is_calibrated_on_test(champion, splits):
    te = splits[2]
    p = champion["pipeline"].predict_proba(te[champion["input_columns"]])[:, 1]
    assert abs(p.mean() - te[P.TARGET].mean()) < 0.03
    t = calibration_table(te[P.TARGET], p)
    assert t.mean_predicted_pd.is_monotonic_increasing and len(t) == 10


def test_coefficient_signs_agree_with_business_sense(champion):
    sc = standardized_coefficients(champion["pipeline"], champion["input_columns"]).groupby("feature").coef.sum()
    for f, c in sc.items():
        if f in EXPECTED_SIGN:
            assert np.sign(c) == EXPECTED_SIGN[f], f"{f} has a counter-intuitive coefficient {c:+.3f}"


def test_reason_codes_are_top3_adverse_and_add_up(champion, splits):
    te = splits[2].head(40)
    feats = champion["input_columns"]
    codes = reason_codes(champion["pipeline"], te, feats, 3)
    assert all(1 <= len(c) <= 3 for c in codes)
    assert all(x["contribution"] > 0 for c in codes for x in c)
    assert all(c[i]["contribution"] >= c[i + 1]["contribution"] for c in codes for i in range(len(c) - 1))
    # decomposition identity: intercept + sum of contributions == log-odds from the model
    contrib = contribution_matrix(champion["pipeline"], te, feats)
    logit = contrib.sum(axis=1) + champion["pipeline"].named_steps["clf"].intercept_[0]
    assert np.allclose(logit, champion["pipeline"].decision_function(te[feats]), atol=1e-6)


def test_encoded_names_map_back_to_features():
    feats = ["employment_type", "log_income"]
    assert encoded_to_feature("nom__employment_type_self_employed", feats) == "employment_type"
    assert encoded_to_feature("num__log_income", feats) == "log_income"


def test_bands_match_params_file():
    prm = load_params()["model"]
    assert P.REFER_CUTOFF == prm["refer_cutoff"] and P.DECLINE_CUTOFF == prm["decline_cutoff"]
    assert [P.risk_level(x) for x in (0.0, 0.10, 0.2199, 0.22, 0.9)] == ["LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]


def test_missing_income_is_scored_like_training(champion, splits):
    """A request without annual_income must give the SAME score the training pipeline gives a NaN income row."""
    te = splits[2]
    row = te[te.annual_income.isna()].iloc[0]
    def py(v):
        return v if isinstance(v, str) else None if np.isnan(v) else v.item()
    fields = {k: py(row[k]) for k in P.PredictRequest.model_fields}
    req = P.PredictRequest(**fields)
    via_api_path = champion["pipeline"].predict_proba(P.request_to_frame(req)[champion["input_columns"]])[:, 1][0]
    via_training_path = champion["pipeline"].predict_proba(te.loc[[row.name], champion["input_columns"]])[:, 1][0]
    assert via_api_path == pytest.approx(via_training_path, abs=1e-9)
    assert P.request_to_frame(req).income_missing.iloc[0] == 1
