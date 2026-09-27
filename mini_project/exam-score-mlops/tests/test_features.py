"""
tests/test_features.py

FEATURE TESTS (PHASE 9): confirm FeatureCreator does what it claims, produces the columns
downstream steps expect, and -- critically -- never leaks information from other rows
(each row's engineered features must depend ONLY on that row's own raw values).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.features import FeatureCreator


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame([
        {"study_hours": 10.0, "attendance_pct": 80.0, "enrollment_date": "2025-01-01"},
        {"study_hours": 5.0, "attendance_pct": 50.0, "enrollment_date": "2025-06-15"},
    ])


def test_feature_creator_is_stateless_fit_returns_self(sample_raw_df):
    fc = FeatureCreator()
    assert fc.fit(sample_raw_df) is fc


def test_creates_expected_new_columns(sample_raw_df):
    fc = FeatureCreator()
    out = fc.fit_transform(sample_raw_df)
    assert "tenure_days" in out.columns
    assert "study_x_attendance" in out.columns


def test_drops_enrollment_date_after_deriving_tenure(sample_raw_df):
    fc = FeatureCreator()
    out = fc.fit_transform(sample_raw_df)
    assert "enrollment_date" not in out.columns


def test_study_x_attendance_is_the_product(sample_raw_df):
    fc = FeatureCreator()
    out = fc.fit_transform(sample_raw_df)
    assert out.loc[0, "study_x_attendance"] == pytest.approx(10.0 * 80.0)
    assert out.loc[1, "study_x_attendance"] == pytest.approx(5.0 * 50.0)


def test_tenure_days_is_non_negative_and_increases_with_earlier_enrollment(sample_raw_df):
    fc = FeatureCreator()
    out = fc.fit_transform(sample_raw_df)
    assert (out["tenure_days"] >= 0).all()
    # row 0 enrolled earlier (2025-01-01) than row 1 (2025-06-15) -> longer tenure
    assert out.loc[0, "tenure_days"] > out.loc[1, "tenure_days"]


def test_transform_does_not_mutate_the_input_dataframe(sample_raw_df):
    original = sample_raw_df.copy()
    fc = FeatureCreator()
    fc.fit_transform(sample_raw_df)
    pd.testing.assert_frame_equal(sample_raw_df, original)


def test_each_rows_features_are_independent_of_other_rows(sample_raw_df):
    """A per-row transform must not leak information across rows -- e.g. via an
    accidental groupby, sort, or running aggregate. Transforming a single row in
    isolation must give the SAME result as transforming it as part of the full batch."""
    fc = FeatureCreator()
    full_batch_result = fc.fit_transform(sample_raw_df)

    single_row_result = fc.fit_transform(sample_raw_df.iloc[[0]])

    assert full_batch_result.loc[0, "study_x_attendance"] == pytest.approx(
        single_row_result.loc[0, "study_x_attendance"]
    )
    assert full_batch_result.loc[0, "tenure_days"] == single_row_result.loc[0, "tenure_days"]
