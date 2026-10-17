"""Feature engineering: hand-calculated values must equal the code, the formulas must be stateless, the registry must be complete."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import emi, engineer
from src.features.definitions import ENGINEERED, FEATURE_KINDS, FRAME_COLUMNS, LOCATIONS, RAW_ELIGIBLE
from src.features.registry import FeatureNotApproved, check_features_approved, load_registry, load_selected

ROW = dict(annual_income=600_000, monthly_income=50_000, employment_years=2.0, loan_amount=300_000, loan_term_months=24,
           credit_history_years=5.0, bureau_score=700, existing_loans=1, monthly_debt=10_000, number_of_dependents=1,
           employment_type="salaried", education="graduate", location="Mumbai")


def test_emi_hand_calculation():
    """P=300000, n=24, r=0.14/12:  (1+r)^24 = 1.320987  ->  EMI = 300000*0.0116667*1.320987/0.320987 = 14,403.86"""
    assert emi(300_000, 24, 14.0) == pytest.approx(14_403.86, abs=0.01)
    assert emi(100_000, 12, 12.0) == pytest.approx(8_884.88, abs=0.01)      # classic 1 lakh / 12 months / 12% example


def test_engineered_values_match_hand_calculation():
    f = engineer(pd.DataFrame([ROW])).iloc[0]
    assert f.debt_to_income_ratio == pytest.approx(0.2)
    assert f.loan_to_income == pytest.approx(0.5)
    assert f.log_income == pytest.approx(np.log(600_001))
    assert f.emi_estimate == pytest.approx(14_403.86, abs=0.01)
    assert f.emi_to_income == pytest.approx(14_403.86 / 50_000, abs=1e-4)
    assert f.debt_service_ratio == pytest.approx((10_000 + 14_403.86) / 50_000, abs=1e-4)
    assert f.income_missing == 0 and f.bureau_score_missing == 0
    assert f.employment_stability == "1_3y" and f.credit_history_bucket == "established" and f.city_tier == "tier1"


def test_missing_income_is_handled_explicitly():
    row = {**ROW, "annual_income": None, "bureau_score": None}
    f = engineer(pd.DataFrame([row])).iloc[0]
    assert f.income_missing == 1 and f.bureau_score_missing == 1
    assert np.isnan(f.log_income) and np.isnan(f.loan_to_income)          # stays NaN: the Pipeline's median imputer fills it
    assert not np.isnan(f.emi_to_income)                                   # monthly_income is always present


@pytest.mark.parametrize("years,bucket", [(0.5, "lt_1y"), (1.0, "1_3y"), (2.9, "1_3y"), (3.0, "3_7y"), (7.0, "7y_plus"), (30, "7y_plus")])
def test_employment_stability_buckets(years, bucket):
    assert engineer(pd.DataFrame([{**ROW, "employment_years": years}])).employment_stability.iloc[0] == bucket


@pytest.mark.parametrize("years,bucket", [(0.5, "thin"), (1.0, "young"), (3.0, "established"), (8.0, "long")])
def test_credit_history_buckets(years, bucket):
    assert engineer(pd.DataFrame([{**ROW, "credit_history_years": years}])).credit_history_bucket.iloc[0] == bucket


def test_engineer_is_stateless_row_wise():
    """The same row gives the same features alone or inside a big batch -> no hidden dependence on other rows (no skew)."""
    big = pd.DataFrame([{**ROW, "annual_income": 100_000 * (i + 1), "loan_amount": 50_000 * (i + 1)} for i in range(50)])
    alone = engineer(big.iloc[[7]]).reset_index(drop=True)
    in_batch = engineer(big).iloc[[7]].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone, in_batch)


def test_all_locations_have_a_tier():
    tiers = engineer(pd.DataFrame([{**ROW, "location": c} for c in LOCATIONS])).city_tier
    assert tiers.notna().all() and set(tiers) == {"tier1", "tier2", "tier3"}


def test_frame_columns_cover_every_feature_kind_entry():
    assert set(FRAME_COLUMNS) <= set(FEATURE_KINDS)
    frame = engineer(pd.DataFrame([ROW]))
    assert set(FRAME_COLUMNS) <= set(frame.columns)


def test_registry_describes_every_column_with_required_metadata():
    reg = load_registry()
    assert set(FEATURE_KINDS) <= set(reg)
    required = {"name", "definition", "source", "data_type", "owner", "calculation_logic", "prediction_time_availability", "status", "reason", "version"}
    for name, rec in reg.items():
        assert required <= set(rec), name
        assert rec["status"] in ("Approved", "Rejected") and rec["reason"]
    for name, (kind, _) in FEATURE_KINDS.items():
        assert reg[name]["data_type"] == kind, f"registry/definitions disagree on {name}"


def test_registry_rejects_leakage_sensitive_and_governance_features():
    reg = load_registry()
    for f in ["interest_rate_offered", "loan_approved", "days_past_due_after_disbursal", "collections_status",
              "post_disbursal_payment_ratio", "credit_officer_override", "gender", "marital_status", "age", "location"]:
        assert reg[f]["status"] == "Rejected", f
    assert reg["age"]["decision_category"] == "REQUIRES_GOVERNANCE" and reg["location"]["decision_category"] == "REQUIRES_GOVERNANCE"
    assert reg["city_tier"]["status"] == "Approved"


def test_training_guard_refuses_unapproved_features():
    for bad in ("gender", "interest_rate_offered", "age", "location", "not_in_registry"):
        with pytest.raises(FeatureNotApproved):
            check_features_approved(["log_income", bad])
    check_features_approved(["log_income", "debt_to_income_ratio"])          # approved features pass silently


def test_selected_features_are_all_approved_and_not_sensitive():
    sel = load_selected()
    check_features_approved(sel)
    assert 5 <= len(sel) <= 20 and not {"gender", "marital_status", "age", "location"} & set(sel)
