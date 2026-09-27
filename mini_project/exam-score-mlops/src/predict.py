"""
src/predict.py

Shared prediction helpers used by both real-time (api/main.py) and batch
(src/batch_predict.py) inference paths, so "how do we load the model and call .predict()"
is defined in exactly one place.
"""
from pathlib import Path
from typing import Union

import joblib
import pandas as pd

# noqa: required so joblib can resolve the FeatureCreator class when unpickling
from src.features import FeatureCreator  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "exam_score_pipeline.joblib"


def load_pipeline(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict_dataframe(pipeline, df: pd.DataFrame) -> pd.Series:
    """Predicts final_score for every row of a RAW (unprocessed) DataFrame -- the pipeline
    performs all feature engineering internally, so `df` should contain exactly the raw
    input columns the model was trained on, nothing pre-derived."""
    predictions = pipeline.predict(df)
    return pd.Series(predictions, index=df.index, name="predicted_final_score").round(1)
