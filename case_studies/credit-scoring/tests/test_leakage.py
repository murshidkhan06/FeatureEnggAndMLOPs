"""Leakage tests: the cheapest and most valuable tests in a lending project."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tests.conftest import needs_data
from src.features.build_features import engineer, time_split
from src.features.definitions import FRAME_COLUMNS, LEAKAGE
from src.features.leakage_demo import target_encode_apply, target_encode_fit
from src.features.registry import load_registry, load_selected
from src.models.pipeline import build_pipeline


def test_no_leakage_column_reaches_the_model_frame():
    assert not set(LEAKAGE) & set(FRAME_COLUMNS)
    assert not set(LEAKAGE) & set(load_selected())


def test_every_leakage_column_is_rejected_with_a_reason():
    reg = load_registry()
    for c in LEAKAGE:
        assert reg[c]["status"] == "Rejected" and len(reg[c]["reason"]) > 20
        assert reg[c]["prediction_time_availability"].startswith("NO")


def test_time_split_is_ordered_and_applicant_disjoint(small_raw):
    tr, va, te = time_split(engineer(small_raw))
    assert tr.application_date.max() <= va.application_date.min() <= te.application_date.min() or tr.application_date.max() <= te.application_date.min()
    assert not set(tr.applicant_id) & set(te.applicant_id)
    assert not set(tr.applicant_id) & set(va.applicant_id)
    assert not set(va.applicant_id) & set(te.applicant_id)
    assert len(tr) > len(va) > 0 and len(te) > 0


def test_imputer_and_scaler_learn_from_train_only(small_raw):
    """Fit on a toy train set, score a test set with absurd values: the learned median must not move."""
    tr, _, te = time_split(engineer(small_raw))
    feats = ["log_income", "loan_to_income", "debt_to_income_ratio"]
    pipe = build_pipeline(feats).fit(tr[feats], tr["default_12m"])
    learned = pipe.named_steps["prep"].named_transformers_["num"].statistics_
    assert np.allclose(learned, tr[feats].median().to_numpy())
    te_weird = te[feats].copy() * 1000
    pipe.predict_proba(te_weird)                                           # predicting must not change anything that was learned
    assert np.allclose(pipe.named_steps["prep"].named_transformers_["num"].statistics_, learned)


def test_leaky_column_inflates_auc_and_collapses_in_production(small_raw):
    tr, _, te = time_split(engineer(small_raw))
    feats = ["log_income", "loan_to_income", "debt_to_income_ratio", "bureau_score"]
    honest = build_pipeline(feats).fit(tr[feats], tr["default_12m"])
    leaky_feats = feats + ["interest_rate_offered"]
    leaky = build_pipeline(leaky_feats).fit(tr[leaky_feats], tr["default_12m"])
    auc_h = roc_auc_score(te["default_12m"], honest.predict_proba(te[feats])[:, 1])
    auc_l = roc_auc_score(te["default_12m"], leaky.predict_proba(te[leaky_feats])[:, 1])
    assert auc_l > auc_h + 0.10 and auc_l > 0.88
    prod = te.copy()
    prod["interest_rate_offered"] = np.nan                                 # the column does not exist yet in production
    assert roc_auc_score(prod["default_12m"], leaky.predict_proba(prod[leaky_feats])[:, 1]) < auc_l - 0.25


def test_target_encoding_formula_hand_example():
    """Segment A: 4 rows, 2 defaults.  Segment B: 16 rows, 0 defaults.  Global rate = 2/20 = 0.10, m = 20
    A -> (2 + 20*0.10) / (4 + 20) = 0.1667 (shrunk from the raw 0.50 towards the global rate)"""
    df = pd.DataFrame(dict(location=["A"] * 4 + ["B"] * 16, employment_type="x", education="y", loan_term_months=12))
    y = pd.Series([1, 1, 0, 0] + [0] * 16)
    table, g = target_encode_fit(df, y, m=20)
    assert g == 0.10
    assert table["A|x|y|12"] == (2 + 20 * 0.10) / (4 + 20)
    unseen = pd.DataFrame(dict(location=["C"], employment_type="x", education="y", loan_term_months=12))
    assert target_encode_apply(unseen, table, g)[0] == g                   # unseen segment -> global rate


@needs_data
def test_processed_splits_are_time_ordered_and_applicant_disjoint(splits):
    tr, va, te = splits
    assert tr.application_date.max() <= te.application_date.min() and va.application_date.max() <= te.application_date.min()
    assert not set(tr.applicant_id) & set(te.applicant_id)
