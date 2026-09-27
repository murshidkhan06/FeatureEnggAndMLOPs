"""
src/data_processing.py

Loading, validation, and splitting -- kept separate from feature engineering (features.py)
and modeling (train.py) so each concern can be tested and reasoned about independently.

WHY validate before anything else touches the data? If a column is silently missing, or
`final_score` sneaks in among the features, or a "numeric" column got read as text, you want
to fail LOUDLY and IMMEDIATELY -- not three steps later with a confusing sklearn error, and
never silently, which is how a bad dataset makes it all the way into a production model.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_RAW_COLUMNS = [
    "student_id", "study_hours", "attendance_pct", "mock_test_1", "mock_test_2",
    "mock_test_3", "income_bracket", "city", "enrollment_date", "shoe_size",
    "lucky_number", "final_score",
]

NUMERIC_COLUMNS = [
    "study_hours", "attendance_pct", "mock_test_1", "mock_test_2", "mock_test_3",
    "shoe_size", "lucky_number", "final_score",
]

CATEGORICAL_COLUMNS = ["income_bracket", "city"]


class DataValidationError(ValueError):
    """Raised when the raw dataset doesn't look like what the pipeline expects."""


def validate_raw_data(df: pd.DataFrame) -> None:
    """Fail loudly on structural problems, before any feature engineering runs.

    Deliberately does NOT reject missing `study_hours` values -- those are expected and
    are exactly what the pipeline's imputer is there to handle. It rejects things that
    indicate a genuinely different or corrupted dataset.
    """
    missing_cols = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {sorted(missing_cols)}")

    if len(df) == 0:
        raise DataValidationError("Dataset is empty.")

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"Column '{col}' should be numeric but has dtype {df[col].dtype}."
            )

    # final_score is the target: it should never be missing in training data (unlike
    # study_hours, which is allowed and expected to have missing values).
    if df["final_score"].isna().any():
        raise DataValidationError("Target column 'final_score' contains missing values.")

    if not df["final_score"].between(0, 100).all():
        raise DataValidationError("'final_score' has values outside the expected 0-100 range.")

    valid_brackets = {"Low", "Medium", "High"}
    unexpected = set(df["income_bracket"].dropna().unique()) - valid_brackets
    if unexpected:
        raise DataValidationError(f"Unexpected income_bracket values: {unexpected}")


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw dataset. This is the ONE place raw data enters the
    pipeline -- train.py, drift.py, and retrain.py all go through this function so
    validation can never accidentally be skipped in one of them."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python data/raw/generate_data.py` "
            f"or `dvc pull` / `dvc checkout` to restore the tracked dataset."
        )
    df = pd.read_csv(path, parse_dates=["enrollment_date"])
    validate_raw_data(df)
    return df


def split_features_target(
    df: pd.DataFrame, target_column: str, id_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drops the id column (never a real predictive feature) and separates X from y."""
    X = df.drop(columns=[id_column, target_column])
    y = df[target_column]
    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
):
    """Thin wrapper kept here (rather than called ad hoc in train.py) so there is exactly
    ONE place that decides how splitting happens -- the single most important step for
    avoiding train/test leakage, per the notebook's own Section 2 explanation."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
