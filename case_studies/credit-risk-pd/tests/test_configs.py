"""Config files agree with the code: quality gate, sample request, selected features, params."""
import json

import yaml

from src import project as P
from src.common.config import load_params
from src.common.paths import CONFIGS
from src.features.definitions import MODEL_FEATURES
from src.governance.feature_lifecycle import check_features_approved


def test_sample_request_is_a_valid_request():
    req = P.PredictRequest(**json.loads((CONFIGS / "sample_request.json").read_text()))
    assert req.customer_id == 10 and str(req.features_as_of) == "2023-03-31"


def test_quality_gate_names_real_metrics():
    gate = json.loads((CONFIGS / "quality_gate.json").read_text())
    assert {"roc_auc", "pr_auc", "recall_at_10pct", "lift_at_10pct", "ks"} == set(gate["min"])


def test_selected_features_are_approved_and_a_subset_of_the_candidates():
    sel = yaml.safe_load((CONFIGS / "selected_features.yaml").read_text())["selected"]
    assert set(sel) <= set(MODEL_FEATURES) and check_features_approved(sel) == sel


def test_params_are_consistent():
    p = load_params()
    assert p["splits"]["train_end"] < p["splits"]["val_start"] < p["splits"]["val_end"] < p["splits"]["test_start"] < p["splits"]["production_start"]
    assert p["retrain"]["train_end"] < p["retrain"]["holdout_start"] <= p["retrain"]["holdout_end"] <= p["splits"]["last_as_of"]
    assert p["drift"]["min_key_features"] >= 2 and set(p["drift"]["key_features"]) <= set(P.DRIFT_FEATURES)


def test_drift_features_are_model_inputs_and_in_the_reference_file():
    assert set(P.DRIFT_FEATURES) <= set(P.INPUT_COLUMNS) and set(P.DRIFT_FEATURES) <= set(P.PredictRequest.model_fields)
