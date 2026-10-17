"""Model tests: metrics worked examples, baselines, pipelines, champion artifact + business sanity.  (Champion tests need `python -m src.models.train`.)"""
import numpy as np
import pandas as pd
import pytest

from src.common.artifact import load_artifact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH
from src.features.registry import check_features_approved
from src.models import metrics as M
from src.models.baselines import BASELINES, baseline_forecast
from src.models.pipelines import build_pipeline
from src.models.train import interval_quantiles, coverage, score, time_split

needs_champion = pytest.mark.skipif(not CHAMPION_PATH.exists(), reason="train a model first")


def test_metric_worked_examples():
    y, f = np.array([100.0, 0.0, 50.0]), np.array([90.0, 10.0, 55.0])
    assert M.mae(y, f) == pytest.approx(25 / 3)
    assert M.wape(y, f) == pytest.approx(25 / 150)                       # sum|e| / sum y
    assert M.bias(y, f) == pytest.approx(5 / 3)                          # mean(f - y): positive = over-forecast
    assert M.bias_pct(y, f) == pytest.approx(5 / 150)
    assert M.rmse(y, f) == pytest.approx(np.sqrt((100 + 100 + 25) / 3))


def test_wape_is_finite_where_mape_explodes():
    y, f = np.array([100.0, 0.0, 50.0]), np.array([90.0, 10.0, 55.0])
    assert not np.isfinite(M.mape_naive(y, f))
    assert np.isfinite(M.wape(y, f))


def test_cost_report_counts_units_by_direction():
    c = M.cost_report([10, 10], [14, 7], over=1.0, under=3.0)
    assert c["over_forecast_units"] == 4 and c["under_forecast_units"] == 3 and c["weighted_cost"] == 4 + 9


def test_baselines_are_just_lags(train_frame):
    assert set(BASELINES) >= {"naive", "seasonal_naive", "moving_average_7"}
    f = train_frame.head(50)
    assert (baseline_forecast("naive", f).values == f["lag_1"].values).all()
    assert (baseline_forecast("seasonal_naive", f).values == f["lag_7"].values).all()


def test_interval_quantiles_and_coverage():
    rng = np.random.default_rng(0)
    f = rng.uniform(50, 150, 5000)
    y = f * np.exp(rng.normal(0, 0.1, 5000))
    q = interval_quantiles(y, f, 0.1, 0.9)
    assert q["low"] < 0 < q["high"] and 0.75 < coverage(y, f, q) < 0.85       # ~80% coverage by construction


def test_both_pipelines_fit_and_predict_non_negative(train_frame):
    p = load_params()
    p["hgb"]["max_iter"] = 30
    feats = ["product_id", "store_id", "day_of_week", "is_holiday", "lag_1", "lag_7", "rolling_mean_7", "promotion_flag", "forecast_temp"]
    sub = train_frame.sample(3000, random_state=0)
    for kind in ("ridge_log", "hgb"):
        pipe = build_pipeline(kind, feats, p).fit(sub[feats], sub["sales"])
        pred = score(pipe, sub, feats)
        assert (pred >= 0).all() and M.wape(sub["sales"], pred) < 0.35


def test_pipeline_handles_missing_forecast_temp(train_frame):
    p = load_params()
    p["hgb"]["max_iter"] = 20
    feats = ["product_id", "lag_1", "rolling_mean_7", "forecast_temp"]
    sub = train_frame.sample(2000, random_state=1)
    pipe = build_pipeline("ridge_log", feats, p).fit(sub[feats], sub["sales"])
    row = sub.head(1).copy()
    row["forecast_temp"] = np.nan
    assert np.isfinite(score(pipe, row, feats)[0])                          # imputed inside the pipeline, no crash


@needs_champion
def test_champion_artifact_contract():
    art = load_artifact(CHAMPION_PATH)
    assert {"pipeline", "version", "input_columns", "metrics", "extra"} <= set(art)
    for k in ("roc_auc", "wape", "bias_pct", "mae", "val_wape"):
        assert k in art["metrics"]
    check_features_approved(art["input_columns"])                           # the gate also protects the artifact
    assert {"low", "high"} <= set(art["extra"]["interval"])


@needs_champion
def test_champion_beats_every_baseline_on_the_test_period(train_frame):
    art = load_artifact(CHAMPION_PATH)
    _, _, te = time_split(train_frame, load_params()["split"])
    champ = M.wape(te["sales"], score(art["pipeline"], te, art["input_columns"]))
    base = {b: M.wape(te["sales"], baseline_forecast(b, te)) for b in BASELINES}
    assert champ < min(base.values()) * 0.85, (champ, base)                  # at least 15% fewer errors than the best baseline
    assert abs(M.bias_pct(te["sales"], score(art["pipeline"], te, art["input_columns"]))) < 0.05


@needs_champion
def test_quality_gate_file_is_respected():
    import json
    from src.common.paths import CONFIGS
    gate = json.loads((CONFIGS / "quality_gate.json").read_text())
    m = load_artifact(CHAMPION_PATH)["metrics"]
    assert all(m[k] >= v for k, v in gate["min"].items())
    assert all(m[k] <= v for k, v in gate["max"].items())
    assert all(abs(m[k]) <= v for k, v in gate["max_abs"].items())
