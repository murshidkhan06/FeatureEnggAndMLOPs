"""Metrics (hand-checkable), pipelines (no leakage through preprocessing), and the trained champion artifact."""
import numpy as np
import pandas as pd
import pytest

from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH
from src.features.definitions import MODEL_FEATURES, TARGET
from src.models.evaluate import business_summary, compute_metrics, decile_table, ks_score, lift_at_top, recall_at_top, top_k_index
from src.models.pipelines import make_hgb, make_logistic

from .conftest import needs_champion, needs_snapshots

SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
LABELS = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])


def test_metric_worked_example_from_the_docstring():
    assert list(top_k_index(SCORES, 0.10)) == [0]
    assert recall_at_top(LABELS, SCORES, 0.10) == pytest.approx(0.5)
    assert lift_at_top(LABELS, SCORES, 0.10) == pytest.approx(5.0)          # precision 1.0 / base rate 0.2
    assert 0.0 <= ks_score(LABELS, SCORES) <= 1.0


def test_compute_metrics_and_business_summary():
    m = compute_metrics(LABELS, SCORES)
    assert m["roc_auc"] == pytest.approx(0.9375)  # 15 of 16 (defaulter, non-defaulter) pairs are ranked correctly and m["recall_at_10pct"] == 0.5 and m["base_rate"] == 0.2
    b = business_summary(LABELS, SCORES, 0.10)
    assert (b["customers_contacted"], b["defaulters_reached"], b["defaulters_missed"]) == (1, 1, 1)
    assert b["calls_per_defaulter_if_random"] == 5.0


def test_perfect_and_random_scores():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    assert compute_metrics(y, y.astype(float))["roc_auc"] == 1.0
    assert abs(compute_metrics(y, np.arange(10) % 3 + 0.0)["roc_auc"] - 0.5) < 0.4


def test_decile_table_is_monotone_for_a_good_ranking():
    rng = np.random.default_rng(0)
    s = rng.random(2000)
    y = (rng.random(2000) < s ** 3).astype(int)
    t = decile_table(y, s)
    assert t["bad_rate"].iloc[0] > t["bad_rate"].iloc[-1] and t["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_pipelines_learn_imputation_inside_the_pipeline():
    """Median imputation + missing indicator + scaler are fitted INSIDE the Pipeline (train only): predicting with NaN works."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    X.loc[::7, "b"] = np.nan
    y = (X["a"].fillna(0) + rng.normal(0, 0.5, 300) > 0.5).astype(int)
    p = {"C": 0.5, "max_iter": 500}
    lr = make_logistic(["a", "b", "c"], p).fit(X, y)
    assert lr.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_[1] == pytest.approx(X["b"].median())
    assert lr.predict_proba(X.head(5).assign(b=np.nan)).shape == (5, 2)
    hgb = make_hgb(["a", "b", "c"], {"max_iter": 20, "random_state": 0}).fit(X, y)
    assert hgb.predict_proba(X.head(5).assign(b=np.nan)).shape == (5, 2)             # trees handle NaN natively


@needs_champion
def test_champion_artifact_contract():
    art = load_artifact(CHAMPION_PATH)
    assert {"roc_auc", "pr_auc", "recall_at_10pct", "lift_at_10pct", "ks"} <= set(art["metrics"])
    assert set(art["input_columns"]) <= set(MODEL_FEATURES) and art["version"] and 0 < art["threshold"] < 1
    assert art["metrics"]["roc_auc"] > 0.75


@needs_champion
@needs_snapshots
def test_champion_beats_a_trivial_baseline_on_the_test_period(snapshots):
    art = load_artifact(CHAMPION_PATH)
    te = snapshots[snapshots["split"] == "test"]
    p = art["pipeline"].predict_proba(te[art["input_columns"]])[:, 1]
    m = compute_metrics(te[TARGET], p)
    assert m["roc_auc"] > 0.85 and m["lift_at_10pct"] > 4 and m["recall_at_10pct"] > 0.5
    naive = compute_metrics(te[TARGET], te["average_balance"].rank(ascending=False))            # "poor customers default" rule of thumb
    assert m["roc_auc"] > naive["roc_auc"] + 0.1


@needs_snapshots
def test_time_split_has_no_overlap(snapshots):
    tr, va, te = (snapshots[snapshots["split"] == s]["as_of_date"] for s in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
