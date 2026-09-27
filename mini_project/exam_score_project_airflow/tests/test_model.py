"""
tests/test_model.py

MODEL TESTS (PHASE 9): the saved pipeline loads, trains successfully from scratch,
produces correctly-shaped and finite predictions, meets a basic performance floor, behaves
sensibly on edge cases (missing values, unseen categories), and is deterministic.

Kept separate from tests/test_api.py (which tests the web layer) so a bug in the ML
pipeline and a bug in the FastAPI layer are never confused for one another.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_processing import load_raw_data, split_features_target, train_test_split_data
from src.evaluate import evaluate_pipeline
from src.pipeline_builder import build_pipeline
from src.predict import load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "student_exam_scores.csv"
MODEL_PATH = REPO_ROOT / "models" / "exam_score_pipeline.joblib"


@pytest.fixture(scope="module")
def params():
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def saved_pipeline():
    if not MODEL_PATH.exists():
        pytest.skip(f"No trained model at {MODEL_PATH} -- run `python -m src.train` first.")
    return load_pipeline(MODEL_PATH)


@pytest.fixture
def sample_student():
    return pd.DataFrame([{
        "study_hours": 12.5,
        "attendance_pct": 88.0,
        "mock_test_1": 72.0,
        "mock_test_2": 75.0,
        "mock_test_3": 70.0,
        "income_bracket": "Medium",
        "city": "Pune",
        "enrollment_date": "2025-06-01",
        "shoe_size": 9.0,
        "lucky_number": 42,
    }])


# ---- "the model can train" -----------------------------------------------------------

def test_pipeline_can_be_built_and_trained_from_scratch(params):
    if not DATA_PATH.exists():
        pytest.skip(f"No dataset at {DATA_PATH} -- run `python data/raw/generate_data.py` first.")

    df = load_raw_data(DATA_PATH)
    X, y = split_features_target(df, params["data"]["target_column"], params["data"]["id_column"])
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, params["split"]["test_size"], params["split"]["random_state"]
    )

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)  # must not raise

    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    assert metrics["mae"] < params["evaluation"]["max_acceptable_mae"], (
        f"MAE {metrics['mae']:.2f} exceeds the acceptable threshold "
        f"{params['evaluation']['max_acceptable_mae']} -- this is a real performance gate, "
        f"not just a smoke test."
    )


# ---- shape / finiteness / determinism --------------------------------------------------

def test_pipeline_loads_successfully(saved_pipeline):
    assert saved_pipeline is not None


def test_prediction_shape_matches_input_rows(saved_pipeline, sample_student):
    two_rows = pd.concat([sample_student, sample_student], ignore_index=True)
    predictions = saved_pipeline.predict(two_rows)
    assert predictions.shape == (2,)


def test_predictions_are_finite(saved_pipeline, sample_student):
    prediction = saved_pipeline.predict(sample_student)
    assert np.isfinite(prediction).all()


def test_pipeline_output_is_deterministic(saved_pipeline, sample_student):
    pred_1 = saved_pipeline.predict(sample_student)[0]
    pred_2 = saved_pipeline.predict(sample_student)[0]
    assert pred_1 == pred_2


# ---- sane predictions / performance floor ----------------------------------------------

def test_pipeline_predicts_a_reasonable_score(saved_pipeline, sample_student):
    prediction = saved_pipeline.predict(sample_student)[0]
    assert 0 <= prediction <= 110  # small headroom -- Linear Regression can slightly overshoot


def test_higher_study_hours_predicts_higher_score(saved_pipeline, sample_student):
    """Directional sanity check: study_hours has a positive coefficient, so all else
    equal, more study hours should never predict a LOWER score."""
    low_effort = sample_student.copy()
    low_effort["study_hours"] = 2.0
    high_effort = sample_student.copy()
    high_effort["study_hours"] = 20.0

    assert saved_pipeline.predict(high_effort)[0] > saved_pipeline.predict(low_effort)[0]


def test_irrelevant_columns_do_not_change_prediction_much(saved_pipeline, sample_student):
    """shoe_size and lucky_number were deliberately dropped by feature selection --
    changing them should NOT meaningfully change the prediction."""
    baseline = saved_pipeline.predict(sample_student)[0]
    changed = sample_student.copy()
    changed["shoe_size"] = 20.0
    changed["lucky_number"] = 999
    assert abs(baseline - saved_pipeline.predict(changed)[0]) < 0.01


# ---- edge cases -------------------------------------------------------------------------

def test_pipeline_handles_missing_study_hours(saved_pipeline, sample_student):
    student = sample_student.copy()
    student["study_hours"] = None
    prediction = saved_pipeline.predict(student)[0]
    assert 0 <= prediction <= 110


def test_pipeline_handles_unseen_city(saved_pipeline, sample_student):
    """A city never seen during training should NOT crash the pipeline -- the one-hot
    encoder was built with handle_unknown='ignore' specifically for this."""
    student = sample_student.copy()
    student["city"] = "Chennai"
    prediction = saved_pipeline.predict(student)[0]
    assert 0 <= prediction <= 110
