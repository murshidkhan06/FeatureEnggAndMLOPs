"""Build the sklearn Pipeline: preprocessing + model in ONE object (so training and serving cannot disagree).

    raw feature frame -> ColumnTransformer -> StandardScaler -> model
                          |- numeric/binary : median imputation          (median learned on TRAIN only)
                          |- ordinal        : ordered categories -> 0,1,2,...
                          |- nominal        : one-hot (unknown category -> all zeros)
    StandardScaler puts every column on the same scale (mean 0, std 1) so coefficients are comparable:
    a "standardized coefficient" says how much the log-odds of default move per 1-standard-deviation change.

Tiny example of median imputation: train values [10, 20, 90, NaN] -> median(10,20,90) = 20 -> NaN becomes 20.
Fitting the imputer on ALL data (including test) would leak test information into the model.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.definitions import categories_of, kind_of


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    num = [f for f in features if kind_of(f) in ("numeric", "binary")]
    ordc = [f for f in features if kind_of(f) == "ordinal"]
    nom = [f for f in features if kind_of(f) == "nominal"]
    parts = []
    if num:
        parts.append(("num", SimpleImputer(strategy="median"), num))
    if ordc:
        parts.append(("ord", OrdinalEncoder(categories=[categories_of(f) for f in ordc],
                                            handle_unknown="use_encoded_value", unknown_value=-1), ordc))
    if nom:
        parts.append(("nom", OneHotEncoder(categories=[categories_of(f) for f in nom], handle_unknown="ignore",
                                           sparse_output=False), nom))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)


def build_pipeline(features: list[str], model: str = "logreg", C: float = 0.5, seed: int = 42) -> Pipeline:
    """model='logreg' -> scorecard-style logistic regression;  model='hgb' -> HistGradientBoosting benchmark."""
    if model == "logreg":
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    elif model == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    else:
        raise ValueError(model)
    return Pipeline([("prep", build_preprocessor(features)), ("scale", StandardScaler()), ("clf", clf)])


def encoded_to_feature(encoded_name: str, features: list[str]) -> str:
    """Map 'nom__employment_type_gig' back to the original feature 'employment_type'."""
    body = encoded_name.split("__", 1)[1]
    if body in features:
        return body
    for f in sorted(features, key=len, reverse=True):
        if body.startswith(f + "_"):
            return f
    raise KeyError(encoded_name)


TARGET = "default_12m"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """data/processed/{train,valid,test}.csv written by `python -m src.features.build_features` (time-based split)."""
    from src.common.paths import DATA_PROCESSED
    return tuple(pd.read_csv(DATA_PROCESSED / f"{n}.csv") for n in ("train", "valid", "test"))  # type: ignore[return-value]


def fit_score(features: list[str], model: str, C: float, train, valid, test, sample_weight=None) -> tuple:
    """Fit on TRAIN only; score valid and test.  Returns (pipeline, metrics_valid, metrics_test)."""
    from src.models.metrics import credit_metrics
    pipe = build_pipeline(features, model=model, C=C)
    fit_kw = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    pipe.fit(train[features], train[TARGET], **fit_kw)
    mv = credit_metrics(valid[TARGET], pipe.predict_proba(valid[features])[:, 1])
    mt = credit_metrics(test[TARGET], pipe.predict_proba(test[features])[:, 1])
    return pipe, mv, mt
