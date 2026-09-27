"""
tests/test_data.py

DATA TESTS (PHASE 9): confirm the raw dataset itself looks the way the rest of the
pipeline assumes it does -- expected columns exist, target exists, dtypes are sane, values
fall in reasonable ranges, and missingness is where (and only where) it's expected to be.

These tests would catch "someone regenerated the dataset with a bug" or "the wrong CSV got
committed" BEFORE that bad data ever reaches feature engineering or training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.data_processing import (
    REQUIRED_RAW_COLUMNS,
    DataValidationError,
    load_raw_data,
    validate_raw_data,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "student_exam_scores.csv"


@pytest.fixture(scope="module")
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip(f"No dataset at {DATA_PATH} -- run `python data/raw/generate_data.py` first.")
    return load_raw_data(DATA_PATH)


def test_dataset_file_exists():
    assert DATA_PATH.exists(), "Run `python data/raw/generate_data.py` to generate it."


def test_all_required_columns_present(raw_df):
    assert set(REQUIRED_RAW_COLUMNS).issubset(set(raw_df.columns))


def test_target_column_has_no_missing_values(raw_df):
    assert raw_df["final_score"].isna().sum() == 0


def test_target_is_within_expected_range(raw_df):
    assert raw_df["final_score"].between(0, 100).all()


def test_study_hours_is_allowed_to_have_missing_values(raw_df):
    # study_hours is the ONE column expected to have real-world missingness -- this is a
    # deliberate property of the dataset, not a bug, and the pipeline's imputer relies on it.
    assert raw_df["study_hours"].isna().sum() > 0


def test_income_bracket_has_only_expected_categories(raw_df):
    assert set(raw_df["income_bracket"].unique()) <= {"Low", "Medium", "High"}


def test_no_duplicate_student_ids(raw_df):
    assert raw_df["student_id"].is_unique


def test_validate_raw_data_rejects_missing_columns():
    bad_df = pd.DataFrame({"study_hours": [1, 2, 3]})
    with pytest.raises(DataValidationError):
        validate_raw_data(bad_df)


def test_validate_raw_data_rejects_target_out_of_range(raw_df):
    bad_df = raw_df.copy()
    bad_df.loc[0, "final_score"] = 999.0
    with pytest.raises(DataValidationError):
        validate_raw_data(bad_df)


def test_validate_raw_data_rejects_missing_target(raw_df):
    bad_df = raw_df.copy()
    bad_df.loc[0, "final_score"] = None
    with pytest.raises(DataValidationError):
        validate_raw_data(bad_df)
