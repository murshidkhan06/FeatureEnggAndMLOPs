"""
src/features.py

WHY does this file exist, separate from the notebook or the training script?
    When you `joblib.dump()` a pipeline that contains a CUSTOM class (like the
    FeatureCreator below), joblib doesn't save the class's actual code -- it only saves a
    REFERENCE to where that class lives (its module path). If that class were defined
    inline in a notebook or a one-off script, its "module path" would be a temporary
    __main__ namespace that doesn't exist anymore once that process ends. Any OTHER
    process trying to load the pickle later (a test script, the FastAPI app, a teammate's
    machine, a Docker container) would then fail with:

        AttributeError: Can't get attribute 'FeatureCreator' on <module '__main__'>

    The fix: define custom classes in a real, importable module -- like this one -- that
    every consumer (training script, notebook, FastAPI app, tests, batch predictor) can
    import from the SAME place, `src.features`. This is one of the most common real
    "it worked in my notebook but broke in production" mistakes, and it is exactly why
    this module is kept separate and never duplicated.

    This class is REUSED VERBATIM (same logic) from the original teaching notebook's
    features.py -- deliberately not redesigned, per the project's feature engineering
    behaviour being preserved end to end.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Creates tenure_days and study_x_attendance from raw columns.
    No fitting required -- every value here is computed independently, per row."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_days"] = (
            pd.Timestamp("2026-01-01") - pd.to_datetime(X["enrollment_date"])
        ).dt.days
        X["study_x_attendance"] = X["study_hours"] * X["attendance_pct"]
        X = X.drop(columns=["enrollment_date"])
        return X
