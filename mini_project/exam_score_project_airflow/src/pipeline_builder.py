"""
src/pipeline_builder.py

Builds the ONE sklearn Pipeline used everywhere -- training, the FastAPI app (indirectly,
via the saved joblib artifact), batch prediction, and retraining. This is the single most
important file for avoiding training-serving skew: every consumer of this project uses the
SAME feature creation -> preprocessing -> PCA -> feature selection -> model pipeline,
because they all load the ONE artifact this function produces, never reimplementing any
step by hand.

The structure below is REUSED, not redesigned, from the original teaching notebook
(mini_project/exam_score_project/notebooks/exam_score_pipeline.ipynb, sections 3-8):
    FeatureCreator -> ColumnTransformer(numeric, mock_tests+PCA, nominal, ordinal)
                    -> SelectFromModel(Lasso)
                    -> LinearRegression
The only change is that every hyperparameter now comes from params.yaml instead of being
hardcoded, so training is reproducible and configurable without touching code.
"""
from typing import Any, Dict

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features import FeatureCreator

# Columns produced by FeatureCreator + the original numeric raw columns. Kept as a module
# constant (rather than re-derived at runtime) so every consumer agrees on what "numeric"
# means without re-running feature creation just to find out.
NUMERIC_COLUMNS = [
    "study_hours", "attendance_pct", "shoe_size", "lucky_number",
    "tenure_days", "study_x_attendance",
]


def build_preprocessor(params: Dict[str, Any]) -> ColumnTransformer:
    prep = params["preprocessing"]
    pca_cfg = params["pca"]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=prep["numeric_impute_strategy"])),
        ("scale", StandardScaler()),
    ])

    mock_test_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=prep["numeric_impute_strategy"])),
        ("scale", StandardScaler()),  # scaling ALWAYS happens before PCA
        ("pca", PCA(n_components=pca_cfg["n_components"], random_state=pca_cfg["random_state"])),
    ])

    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # unseen category -> all zeros, no crash
    ])

    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[prep["ordinal_categories"]])),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_COLUMNS),
        ("mock_tests_pca", mock_test_pipe, prep["mock_test_columns"]),
        ("nominal", nominal_pipe, prep["nominal_columns"]),
        ("ordinal", ordinal_pipe, prep["ordinal_columns"]),
    ])


def build_pipeline(params: Dict[str, Any]) -> Pipeline:
    """Builds the full, unfitted, leakage-safe pipeline. Call `.fit(X_train, y_train)` on
    the result -- never fit any step of this pipeline individually outside of it."""
    fs_cfg = params["feature_selection"]

    preprocessor = build_preprocessor(params)
    feature_selector = SelectFromModel(
        Lasso(alpha=fs_cfg["lasso_alpha"], random_state=fs_cfg["random_state"])
    )

    return Pipeline([
        ("feature_creation", FeatureCreator()),
        ("preprocessing", preprocessor),
        ("feature_selection", feature_selector),
        ("model", LinearRegression()),
    ])
