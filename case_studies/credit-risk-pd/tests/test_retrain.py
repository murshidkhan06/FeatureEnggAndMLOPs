"""Retraining workflow: windows, label maturity, promotion + rollback in a sandbox, and the human-approval rule."""
import json
import subprocess
import sys

import joblib
import pandas as pd
import pytest

from src.common import promotion
from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, ROOT
from src.models import retrain as R

from .conftest import needs_champion, needs_production, needs_snapshots


def test_next_version():
    assert R.next_version("v1") == "v2" and R.next_version("v9") == "v10" and R.next_version("prod") == "prod-next"


def test_months_between():
    assert R._months_between("2024-04-30", "2024-10-31") == 6


@needs_snapshots
def test_label_maturity_and_windows(snapshots):
    rp = {"train_end": "2024-04-30", "stress_window_start": "2023-09-30", "today": "2025-06-30"}
    assert R.mature(snapshots, "2025-06-30")["as_of_date"].max() == pd.Timestamp("2024-12-31")
    assert R.mature(snapshots, "2024-02-29")["as_of_date"].max() == pd.Timestamp("2023-08-31")         # 6-month label lag
    roll, stress = R.training_window(snapshots, rp, "rolling"), R.training_window(snapshots, rp, "stress_only")
    assert roll["as_of_date"].max() == stress["as_of_date"].max() == pd.Timestamp("2024-04-30")
    assert stress["as_of_date"].min() == pd.Timestamp("2023-09-30") and len(stress) < len(roll) / 3
    hold_start = pd.Timestamp("2024-10-31")
    assert roll["as_of_date"].max() + pd.DateOffset(months=6) <= hold_start               # purge gap: window does not touch the hold-out


def _fake_model(path, version, auc):
    joblib.dump(dict(pipeline=None, version=version, input_columns=[], metrics={"roc_auc": auc}, threshold=0.1, extra={}, trained_at="x"), path)


def test_promotion_and_rollback_in_a_sandbox(tmp_path, monkeypatch):
    for name, p in dict(CHAMPION_PATH=tmp_path / "champion.joblib", CHALLENGER_PATH=tmp_path / "challenger.joblib", REGISTRY_PATH=tmp_path / "registry.json").items():
        monkeypatch.setattr(promotion, name, p)
    _fake_model(promotion.CHAMPION_PATH, "v1", 0.90)
    promotion.register_bootstrap("v1", {"roc_auc": 0.90})
    _fake_model(promotion.CHALLENGER_PATH, "v2", 0.92)
    assert promotion.promote_challenger("Priya (Model Risk)", "test") == "v2"
    assert joblib.load(promotion.CHAMPION_PATH)["version"] == "v2" and (tmp_path / "model_v1.joblib").exists()
    hist = json.loads(promotion.REGISTRY_PATH.read_text())
    assert [(h["version"], h["stage"]) for h in hist] == [("v1", "Archived"), ("v2", "Production")] and hist[1]["approved_by"] == "Priya (Model Risk)"
    assert promotion.rollback() == "v1" and joblib.load(promotion.CHAMPION_PATH)["version"] == "v1"
    assert [h["stage"] for h in json.loads(promotion.REGISTRY_PATH.read_text())] == ["Production", "Archived"]


def test_two_promote_rollback_cycles_restore_the_right_model(tmp_path, monkeypatch):
    """Regression (shared-code fix): after a rollback, the NEXT rollback must restore the model that was really replaced, not 'the newest Archived one'."""
    for name, p in dict(CHAMPION_PATH=tmp_path / "champion.joblib", CHALLENGER_PATH=tmp_path / "challenger.joblib", REGISTRY_PATH=tmp_path / "registry.json").items():
        monkeypatch.setattr(promotion, name, p)
    serving = lambda: joblib.load(promotion.CHAMPION_PATH)["version"]
    _fake_model(promotion.CHAMPION_PATH, "v1", 0.90)
    promotion.register_bootstrap("v1", {"roc_auc": 0.90})
    _fake_model(promotion.CHALLENGER_PATH, "v2", 0.91)
    promotion.promote_challenger("A", "cycle 1")                  # v1 -> v2
    assert promotion.rollback() == "v1" and serving() == "v1"     # back to v1
    _fake_model(promotion.CHALLENGER_PATH, "v3", 0.92)
    promotion.promote_challenger("B", "cycle 2")                  # v1 -> v3
    assert serving() == "v3"
    assert promotion.rollback() == "v1" and serving() == "v1"     # v3 replaced v1, so v1 (NOT v2) comes back
    _fake_model(promotion.CHALLENGER_PATH, "v4", 0.93)
    promotion.promote_challenger("C", "cycle 3")
    _fake_model(promotion.CHALLENGER_PATH, "v5", 0.94)
    promotion.promote_challenger("D", "cycle 4")                  # v1 -> v4 -> v5
    assert promotion.rollback() == "v4" and serving() == "v4"
    assert promotion.rollback() == "v1" and serving() == "v1"


def test_compare_lower_is_better_and_absolute_limits():
    champ = {"roc_auc": 0.80, "err": 0.10}
    assert promotion.compare(champ, {"roc_auc": 0.80, "err": 0.08}, primary="err", lower_is_better=["err"], min_improvement=0.01)["promote"]
    assert not promotion.compare(champ, {"roc_auc": 0.85, "err": 0.20}, absolute_limits={"err": 0.15})["promote"]


def test_rollback_without_history_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(promotion, "REGISTRY_PATH", tmp_path / "registry.json")
    assert promotion.rollback() is None


def _run(*args):
    p = subprocess.run([sys.executable, "-m", "src.models.retrain", *args], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    return p.stdout


@needs_champion
@needs_snapshots
@needs_production
def test_end_to_end_without_approval_stops_and_changes_nothing():
    before = load_artifact(CHAMPION_PATH)["version"], (ROOT / "models" / "champion.joblib").stat().st_mtime
    out = _run("--prod", "data/production/batch_drifted.csv")
    assert "RETRAIN_CANDIDATE" in out and "AWAITING APPROVAL" in out and "PROMOTED:" not in out
    assert (load_artifact(CHAMPION_PATH)["version"], (ROOT / "models" / "champion.joblib").stat().st_mtime) == before      # champion untouched


@needs_champion
@needs_snapshots
@needs_production
@pytest.mark.parametrize("batch,expected", [("batch_normal.csv", "STOP: the ladder says MONITOR"), ("batch_dq_issue.csv", "STOP: the ladder says FIX_DATA_PIPELINE")])
def test_ladder_stops_before_training(batch, expected):
    out = _run("--prod", f"data/production/{batch}")
    assert expected in out and "STEP 4" not in out
