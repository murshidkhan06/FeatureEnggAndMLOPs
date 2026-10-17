"""The two sklearn Pipelines (preprocessing + model in ONE object -> no training-serving skew).

WHY a Pipeline
  Imputation values, scaling statistics and model weights are all LEARNED from the training data.  If they were
  computed outside the model object, the API would have to re-implement them (and would drift apart).  Putting
  everything into one Pipeline means  joblib.dump(pipeline)  carries the exact fitted preprocessing to production.

BASELINE  logistic regression
    median imputation (+ missing indicator for columns that had NaN in training) -> StandardScaler -> LogisticRegression
    P(default) = 1 / (1 + exp(-(b0 + b1*x1 + ... )))     x are standardised, so |b| compares feature strengths
    Tiny example: b0 = -3.2, b1 = 0.8 (standardised utilisation), x1 = +2  ->  z = -1.6  ->  p = 1/(1+e^1.6) = 0.168
CANDIDATE HistGradientBoostingClassifier
    a few hundred small decision trees added one after another, each fixing the mistakes of the previous ones.
    Handles NaN natively, learns thresholds ("utilisation above 35 % is dangerous") and interactions without hand-made features.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic(features: list[str], p: dict) -> Pipeline:
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                                ("scale", StandardScaler())]), features)], remainder="drop")
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=p["C"], max_iter=p["max_iter"]))])


def make_hgb(features: list[str], p: dict) -> Pipeline:
    prep = ColumnTransformer([("num", "passthrough", features)], remainder="drop")     # only selects the columns
    return Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(**p))])


MAKERS = {"logistic": make_logistic, "hgb": make_hgb}
