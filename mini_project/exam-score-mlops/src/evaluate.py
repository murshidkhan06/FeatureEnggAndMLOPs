"""
src/evaluate.py

Evaluation logic, kept separate from train.py so the SAME evaluation code can be reused by
retrain.py to compare a candidate model against the currently-registered production model
on identical terms.
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_pipeline(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Returns R2, MAE, and RMSE for a fitted pipeline on held-out data."""
    y_pred = pipeline.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
    }


def feature_coefficients(pipeline) -> pd.DataFrame:
    """Returns the surviving features and their Linear Regression coefficients --
    interpretation only makes sense for a linear model, which is exactly why this project
    deliberately uses one (see the notebook's Section 10)."""
    preprocessor = pipeline.named_steps["preprocessing"]
    selector = pipeline.named_steps["feature_selection"]
    model = pipeline.named_steps["model"]

    all_names = preprocessor.get_feature_names_out()
    kept_names = all_names[selector.get_support()]

    return pd.DataFrame({
        "feature": kept_names,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)
