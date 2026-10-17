"""Model pipelines.  Every preprocessing step lives INSIDE the sklearn Pipeline so it is fitted on
TRAINING rows only and re-applied unchanged at prediction time (no leakage, no skew).

Two ML candidates (plus the baselines in baselines.py):

  ridge_log   Ridge regression on a LOG target.
      Why log?  Demand is MULTIPLICATIVE: a promotion adds +40% to a series that sells 50 units and +40% to one
      that sells 500.  log(sales) turns "x 1.4" into "+ 0.34", which a linear model can add up.
      log1p(x) = log(1 + x) is safe when x = 0 (stock-out).      e.g. log1p(99) = 4.605;  expm1(4.605) = 99
      Ridge = least squares + a penalty alpha * sum(w^2) that shrinks coefficients (less over-fitting
      with correlated lags).   Because it works with log(lag) features it still follows a series whose level moves
      to a range never seen in training (the trend shift) - it EXTRAPOLATES.

  hgb         HistGradientBoostingRegressor (a tree ensemble) on raw units.
      Flexible: learns interactions (promotion x weekend x product) automatically.
      Weakness: a tree predicts an AVERAGE OF TRAINING TARGETS, so it can never output more than it saw in
      training.  If the level of the business jumps above the training range, it under-predicts.
      (Teaching moment of this project.)

Why not an LSTM / Prophet / ARIMA?  The course stack is sklearn; per-series models would need 24 fits and cannot
share what one product teaches another; and a simple pooled model with lag features is a strong, explainable baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.definitions import IDENTITY, LAGS

FLAGS = ["is_weekend", "is_holiday", "promotion_flag"]
MODEL_NAMES = ["ridge_log", "hgb"]


class LogRidge(BaseEstimator, RegressorMixin):
    """Ridge on log1p(sales), with a calibration factor so the forecasts add up to the sales we actually saw.

    Why the factor?  Predicting the average of log(y) gives something close to the MEDIAN of y, which is slightly
    below the MEAN of y (the mean is pulled up by occasional big days).  Without a correction the forecast is
    biased low.  The fix: calibration = sum(actual) / sum(predicted) on the training rows (e.g. 1.02 = +2%).
    Worked example: actual sales 100, 200 -> predictions 98, 196 -> factor = 300 / 294 = 1.0204.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.ridge_ = Ridge(alpha=self.alpha).fit(X, np.log1p(y))
        raw = np.expm1(self.ridge_.predict(X))
        self.calibration_ = float(y.sum() / raw.sum())
        return self

    def predict(self, X):
        return np.clip(np.expm1(self.ridge_.predict(X)) * self.calibration_, 0, None)


def _log1p_transformer() -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_pipeline(kind: str, features: list[str], params: dict) -> Pipeline:
    """kind in {'ridge_log', 'hgb'};  `features` = the selected model input columns."""
    ids = [f for f in IDENTITY if f in features]
    level = [f for f in LAGS if f in features]
    temp = [f for f in ["forecast_temp"] if f in features]
    flags = [f for f in FLAGS if f in features]
    if kind == "ridge_log":
        dow = [f for f in ["day_of_week"] if f in features]
        used = set(ids + dow + level + temp + flags)
        numeric = [f for f in features if f not in used]
        parts = []
        if ids or dow:
            parts.append(("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ids + dow))
        if level:
            parts.append(("log_level", Pipeline([("log", _log1p_transformer()), ("scale", StandardScaler())]), level))
        if temp:   # weather outage -> median + a "was missing" indicator column
            parts.append(("temp", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                            ("scale", StandardScaler())]), temp))
        if numeric:
            parts.append(("numeric", StandardScaler(), numeric))
        if flags:
            parts.append(("flags", "passthrough", flags))
        pre = ColumnTransformer(parts, verbose_feature_names_out=False)
        model = LogRidge(alpha=params["ridge"]["alpha"])
    elif kind == "hgb":
        parts = []
        if ids:
            parts.append(("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ids))
        rest = [f for f in features if f not in ids]
        pre = ColumnTransformer(parts + [("rest", "passthrough", rest)], verbose_feature_names_out=False)
        model = HistGradientBoostingRegressor(**params["hgb"])
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return Pipeline([("prep", pre), ("model", model)])
