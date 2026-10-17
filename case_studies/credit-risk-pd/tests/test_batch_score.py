"""Monthly batch scoring (the realistic production mode)."""
import numpy as np
import pandas as pd
import pytest

from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH
from src.models.batch_score import score_month

from .conftest import needs_champion, needs_snapshots


@needs_champion
def test_batch_scores_everyone_once_and_ranks_them(raw):
    art = load_artifact(CHAMPION_PATH)
    s = score_month("2024-12-31", art, raw, top_pct=0.10)
    assert s["customer_id"].is_unique and s["probability_of_default_6m"].between(0, 1).all()
    assert s["probability_of_default_6m"].is_monotonic_decreasing and s["rank"].tolist() == list(range(1, len(s) + 1))
    assert s["contact_recommended"].sum() == round(0.10 * len(s)) and set(s["risk_band"]) <= {"LOW", "MEDIUM", "HIGH"}
    assert (s["features_as_of"].astype(str) == "2024-12-31").all() and (s["model_version"] == art["version"]).all()


@needs_champion
@needs_snapshots
def test_batch_scores_equal_training_snapshot_scores(raw, snapshots):
    """Offline training table and monthly batch pipeline produce the SAME numbers for the same month (no skew between them)."""
    art = load_artifact(CHAMPION_PATH)
    s = score_month("2022-12-31", art, raw).set_index("customer_id")["probability_of_default_6m"]
    snap = snapshots[snapshots["as_of_date"] == "2022-12-31"].set_index("customer_id")
    expected = pd.Series(art["pipeline"].predict_proba(snap[art["input_columns"]])[:, 1], index=snap.index).round(4)
    assert set(s.index) == set(snap.index)
    assert np.allclose(s.sort_index().to_numpy(), expected.sort_index().to_numpy(), atol=1e-4)


@needs_champion
def test_nobody_already_in_default_is_scored(raw):
    art = load_artifact(CHAMPION_PATH)
    s = score_month("2024-06-30", art, raw)
    d = raw["defaults"]
    gone = set(d.loc[d["default_date"] <= "2024-06-30", "customer_id"])
    assert gone and not (gone & set(s["customer_id"]))
