"""Governance: fairness maths, review gate behaviour, model card content, age/location policy decisions."""
import numpy as np
import pandas as pd
import pytest

from tests.conftest import needs_data, needs_model
from src.features.registry import load_registry
from src.governance import review_gate
from src.governance.fairness_audit import adverse_impact_ratio, age_band, equal_opportunity_gap, group_table


def test_adverse_impact_ratio_hand_example():
    assert adverse_impact_ratio([0.72, 0.80]) == pytest.approx(0.90)          # 0.72 / 0.80
    assert adverse_impact_ratio([0.50, 0.80]) == pytest.approx(0.625)         # fails the four-fifths rule
    assert equal_opportunity_gap([0.88, 0.84]) == pytest.approx(0.04)


def test_group_table_hand_example():
    # group F: 4 applicants, 1 declined (a defaulter);  group M: 4 applicants, 2 declined (one defaulter, one good)
    g = pd.Series(list("FFFFMMMM"))
    y = pd.Series([1, 0, 0, 0, 1, 0, 0, 0])
    declined = np.array([1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    t = group_table(g, y, declined).set_index("group")
    assert t.loc["F", "approval_rate"] == 0.75 and t.loc["M", "approval_rate"] == 0.5
    assert t.loc["F", "approved_among_repaid"] == 1.0 and t.loc["M", "approved_among_repaid"] == pytest.approx(2 / 3, abs=1e-3)
    assert adverse_impact_ratio(t.approval_rate) == pytest.approx(0.6667, abs=1e-3)


def test_age_bands():
    assert list(age_band(pd.Series([21, 25, 34, 35, 49, 50, 65]))) == ["<25", "25-34", "25-34", "35-49", "35-49", "50+", "50+"]


def test_age_and_gender_are_policy_decisions_recorded_in_registry():
    reg = load_registry()
    assert reg["age"]["status"] == "Rejected" and "POLICY" in reg["age"]["reason"] and "differ by country" in reg["age"]["reason"]
    assert reg["gender"]["sensitivity"] == "sensitive_attribute" and reg["marital_status"]["sensitivity"] == "sensitive_attribute"
    assert reg["location"]["sensitivity"] == "proxy_risk" and reg["city_tier"]["status"] == "Approved"


@needs_data
@needs_model
def test_review_gate_passes_the_champion_with_a_complete_card(champion, tmp_path):
    card = tmp_path / "card.md"
    card.write_text("\n".join(f"## {s}" for s in review_gate.REQUIRED_SECTIONS))
    results = review_gate.run_gate(champion, card_path=card)
    assert all(r["passed"] for r in results), [r for r in results if not r["passed"]]


@needs_data
@needs_model
def test_review_gate_fails_on_each_rule(champion, tmp_path, monkeypatch):
    card = tmp_path / "card.md"
    card.write_text("\n".join(f"## {s}" for s in review_gate.REQUIRED_SECTIONS))
    ok = review_gate.run_gate(champion, card_path=card)
    assert all(r["passed"] for r in ok)
    # 1+2: unapproved / sensitive feature in the model
    bad = dict(champion, input_columns=champion["input_columns"] + ["gender"])
    res = review_gate.run_gate(bad, card_path=card)
    assert not res[0]["passed"] and not res[1]["passed"]
    # 3: adverse impact ratio below 0.8
    real_audit = review_gate.run_audit
    monkeypatch.setattr(review_gate, "run_audit", lambda art: dict(real_audit(art), gender_summary=dict(air=0.70), gate_attribute="gender", min_air=0.8))
    assert not review_gate.run_gate(champion, card_path=card)[2]["passed"]
    monkeypatch.setattr(review_gate, "run_audit", real_audit)
    # 4: missing model card
    assert not review_gate.run_gate(champion, card_path=tmp_path / "missing.md")[3]["passed"]
    # 5: AUC below minimum
    assert not review_gate.run_gate(dict(champion, metrics={"roc_auc": 0.60}), card_path=card)[4]["passed"]
    # 6: no reason codes (black-box learner)
    from src.models.pipeline import build_pipeline, load_splits
    tr = load_splits()[0]
    feats = champion["input_columns"]
    hgb = dict(champion, pipeline=build_pipeline(feats, "hgb").fit(tr[feats], tr["default_12m"]))
    assert not review_gate.run_gate(hgb, card_path=card)[5]["passed"]


@needs_data
@needs_model
def test_model_card_contains_all_required_sections():
    from src.governance.model_card import build_card
    text = build_card()
    for s in review_gate.REQUIRED_SECTIONS:
        assert s.lower() in text.lower(), s
    assert "PENDING (human)" in text and "Adverse impact ratio" in text
