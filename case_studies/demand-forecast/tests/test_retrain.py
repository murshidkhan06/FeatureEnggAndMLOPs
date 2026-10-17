"""Retraining workflow tests: windows, idempotent refresh, promotion lifecycle (promote -> rollback), data-quality tests,
and the drift ladder on the real batches."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.common import promotion
from src.common.artifact import load_artifact, save_artifact
from src.common.paths import CHAMPION_PATH, DATA_PRODUCTION, HISTORY_PATH
from src.features import data_quality
from src.features.definitions import HISTORY_COLUMNS
from src.models.retrain import holdout_metrics, make_windows
from src.models.weekly_job import upsert_history


# ---------------------------------------------------------------- windows
def test_windows_are_chronological_and_rolling_is_smaller(train_frame):
    exp = make_windows(train_frame, "expanding", 28, 14, 12)
    rol = make_windows(train_frame, "rolling", 28, 14, 12)
    for w in (exp, rol):
        assert w["train"]["date"].max() < w["valid"]["date"].min() <= w["valid"]["date"].max() < w["hold"]["date"].min()
    assert exp["hold"]["date"].nunique() == 28 and exp["valid"]["date"].nunique() == 14
    assert len(rol["train"]) < len(exp["train"])
    assert rol["train"]["date"].min() >= rol["valid"]["date"].min() - pd.DateOffset(months=12) - pd.Timedelta(days=1)
    assert exp["hold"].equals(rol["hold"])                                    # SAME hold-out for both strategies


def test_unknown_strategy_is_rejected(train_frame):
    with pytest.raises(ValueError):
        make_windows(train_frame, "sliding", 28, 14, 12)


def test_holdout_metrics_have_absolute_bias():
    f = pd.DataFrame(dict(sales=[100.0, 100.0], rolling_mean_30=[90.0, 90.0]))
    m = holdout_metrics(f, np.array([80.0, 90.0]))
    assert m["bias_pct"] == pytest.approx(-0.15) and m["abs_bias_pct"] == pytest.approx(0.15)


# ---------------------------------------------------------------- weekly refresh (idempotent)
def test_upsert_history_is_idempotent(raw):
    h = raw[HISTORY_COLUMNS].iloc[:5000]
    new = raw[HISTORY_COLUMNS].iloc[4900:5100]                                  # 100 rows already exist, 100 are new
    once, info = upsert_history(h, new)
    twice, info2 = upsert_history(once, new)
    assert info == dict(rows_before=5000, new=100, replaced=100, rows_after=5100)
    assert len(twice) == len(once) == 5100 and info2["new"] == 0
    assert not twice.duplicated(["date", "product_id", "store_id"]).any()


# ---------------------------------------------------------------- data tests behind the failure scenarios
def _batch(raw):
    return raw[(raw.date >= "2024-12-01") & (raw.date <= "2024-12-14")].rename(columns={"date": "forecast_date"})


def test_clean_batch_passes_data_quality(raw):
    res = data_quality.validate(_batch(raw), reference=None)
    assert res["ok"], res["problems"]


def test_duplicated_rows_fail_data_quality(raw):
    b = _batch(raw)
    res = data_quality.validate(pd.concat([b, b.tail(24)]), reference=None)
    assert not res["ok"] and any("duplicates" in p for p in res["problems"])


def test_price_unit_change_fails_data_quality(raw):
    b = _batch(raw)
    ref = raw[raw.date < "2024-12-01"][["product_id", "store_id", "price"]]
    res = data_quality.validate(b.assign(price=b["price"] * 100), reference=ref)
    assert not res["ok"] and any("price_unit" in p for p in res["problems"])


def test_missing_column_fails_data_quality(raw):
    res = data_quality.validate(_batch(raw).drop(columns=["price"]), reference=None)
    assert not res["ok"] and "schema" in res["problems"][0]


# ---------------------------------------------------------------- promotion lifecycle on temporary files
def _fake_registry(tmp_path, monkeypatch):
    champ, chall, reg = tmp_path / "champion.joblib", tmp_path / "challenger.joblib", tmp_path / "registry.json"
    monkeypatch.setattr(promotion, "CHAMPION_PATH", champ)
    monkeypatch.setattr(promotion, "CHALLENGER_PATH", chall)
    monkeypatch.setattr(promotion, "REGISTRY_PATH", reg)
    save_artifact(champ, DummyRegressor(constant=1.0), "v1", ["a"], {"wape": 0.2})
    promotion.register_bootstrap("v1", {"wape": 0.2})
    return champ, chall, reg


def test_repeated_promote_and_rollback_cycles_restore_the_right_model(tmp_path, monkeypatch):
    """Regression test: a second cycle must roll back to the model it REPLACED, not to the newest 'Archived' entry."""
    champ, chall, reg = _fake_registry(tmp_path, monkeypatch)
    for new, expect_back in (("v2", "v1"), ("v3", "v1")):
        save_artifact(chall, DummyRegressor(constant=2.0), new, ["a"], {"wape": 0.1})
        promotion.promote_challenger("Asha", "cycle")
        assert load_artifact(champ)["version"] == new
        assert promotion.rollback() == expect_back
        assert load_artifact(champ)["version"] == expect_back
    stages = {h["version"]: h["stage"] for h in json.loads(reg.read_text())}
    assert stages["v1"] == "Production" and stages["v2"] == stages["v3"] == "Archived"


def test_rollback_without_history_returns_none(tmp_path, monkeypatch):
    _fake_registry(tmp_path, monkeypatch)
    assert promotion.rollback() is None


def test_promote_then_rollback(tmp_path, monkeypatch):
    champ, chall, reg = tmp_path / "champion.joblib", tmp_path / "challenger.joblib", tmp_path / "registry.json"
    for mod in (promotion,):
        monkeypatch.setattr(mod, "CHAMPION_PATH", champ)
        monkeypatch.setattr(mod, "CHALLENGER_PATH", chall)
        monkeypatch.setattr(mod, "REGISTRY_PATH", reg)
    save_artifact(champ, DummyRegressor(constant=1.0), "v1", ["a"], {"wape": 0.2})
    promotion.register_bootstrap("v1", {"wape": 0.2})
    save_artifact(chall, DummyRegressor(constant=2.0), "v2", ["a"], {"wape": 0.1})
    assert promotion.promote_challenger("Asha", "test") == "v2"
    assert load_artifact(champ)["version"] == "v2"
    hist = json.loads(reg.read_text())
    assert [h["stage"] for h in hist] == ["Archived", "Production"] and hist[-1]["approved_by"] == "Asha"
    assert promotion.rollback() == "v1"
    assert load_artifact(champ)["version"] == "v1"
    assert [h["stage"] for h in json.loads(reg.read_text())] == ["Production", "Archived"]


# ---------------------------------------------------------------- the ladder on the real batches (needs champion + data)
def _bootstrap_champion_serving() -> bool:
    """The ladder lessons assume the FIRST champion (v1) is serving; after a real promotion the story is different (and that is fine)."""
    return CHAMPION_PATH.exists() and load_artifact(CHAMPION_PATH)["version"] == "v1"


needs_all = pytest.mark.skipif(not (_bootstrap_champion_serving() and HISTORY_PATH.exists() and (DATA_PRODUCTION / "batch_drifted.csv").exists()),
                               reason="generate data and train the bootstrap champion (v1) first")


@needs_all
def test_ladder_drifted_batch_is_a_retrain_candidate_normal_is_not():
    from src.monitoring.run_drift_check import assess
    drifted = assess(str(DATA_PRODUCTION / "batch_drifted.csv"), verbose=False)
    assert drifted["decision"]["decision"] == "RETRAIN_CANDIDATE" and drifted["lag_drift"]
    normal = assess(str(DATA_PRODUCTION / "batch_normal.csv"), verbose=False)
    assert normal["decision"]["decision"] in ("MONITOR", "NO_ACTION")               # drift alone never triggers a retrain


@needs_all
def test_festival_shift_is_drift_without_failure():
    from src.monitoring.run_drift_check import assess
    r = assess(str(DATA_PRODUCTION / "batch_festival.csv"), verbose=False)
    statuses = {t["feature"]: t["status"] for t in r["table"]}
    assert statuses["rolling_mean_7"] == "drift"                                    # inputs shifted ...
    assert r["decision"]["decision"] in ("MONITOR", "NO_ACTION")                    # ... the model still performs: expected seasonal shift
