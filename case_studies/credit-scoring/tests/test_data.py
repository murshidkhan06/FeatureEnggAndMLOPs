"""The synthetic data must contain exactly the real-world problems we promise in docs/DATASET.md."""
import numpy as np
import pandas as pd

from tests.conftest import needs_data
from src.data.make_dataset import AUDIT_ONLY, REQUEST_FIELDS, TARGET, generate
from src.features.definitions import LEAKAGE


def test_generator_is_deterministic():
    a, b, c = generate(1500, seed=5), generate(1500, seed=5), generate(1500, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_target_rate_is_realistic(small_raw):
    assert 0.07 <= small_raw[TARGET].mean() <= 0.14


def test_annual_income_missing_about_12_percent(small_raw):
    assert 0.09 <= small_raw.annual_income.isna().mean() <= 0.15


def test_missingness_is_not_random(small_raw):
    """MNAR-ish: self-employed / business applicants skip annual income more often than salaried ones."""
    miss = small_raw.assign(m=small_raw.annual_income.isna()).groupby("employment_type").m.mean()
    assert miss[["self_employed", "business"]].min() > miss["salaried"]


def test_monthly_income_is_always_present_and_correlated(small_raw):
    assert small_raw.monthly_income.notna().all()
    both = small_raw[["annual_income", "monthly_income"]].dropna()
    assert both.corr(method="spearman").iloc[0, 1] > 0.85          # Spearman: robust to the injected income outliers


def test_debt_and_dti_are_correlated(small_raw):
    assert small_raw[["monthly_debt", "debt_to_income_ratio"]].corr(method="spearman").iloc[0, 1] > 0.75


def test_income_is_right_skewed_with_outliers(small_raw):
    inc = small_raw.annual_income.dropna()
    assert inc.mean() > inc.median() and inc.max() > 5 * inc.median()


def test_leakage_and_sensitive_columns_exist_in_raw(small_raw):
    for c in LEAKAGE + ["gender", "marital_status", "age", "location"]:
        assert c in small_raw.columns
    assert small_raw.location.nunique() == 15


def test_repeat_applicants_exist(small_raw):
    assert small_raw.applicant_id.duplicated().mean() > 0.04


def test_leakage_columns_really_leak(small_raw):
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(small_raw[TARGET], small_raw.interest_rate_offered) > 0.88
    assert roc_auc_score(small_raw[TARGET], small_raw.days_past_due_after_disbursal) > 0.95


@needs_data
def test_production_batches_have_api_columns_target_and_no_leakage(batches):
    for b in batches:
        assert set(REQUEST_FIELDS) <= set(b.columns) and TARGET in b.columns
        assert set(AUDIT_ONLY) <= set(b.columns)
        assert not set(LEAKAGE) & set(b.columns)
        assert len(b) >= 2000


@needs_data
def test_drifted_batch_is_an_economic_downturn(batches):
    normal, drifted = batches
    assert drifted.annual_income.median() < 0.9 * normal.annual_income.median()
    assert (drifted.monthly_debt / drifted.monthly_income).median() > 1.15 * (normal.monthly_debt / normal.monthly_income).median()
    assert drifted[TARGET].mean() > normal[TARGET].mean() + 0.03
