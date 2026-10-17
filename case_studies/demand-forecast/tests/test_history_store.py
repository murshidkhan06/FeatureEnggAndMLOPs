"""ONLINE vs OFFLINE features: the history store and training must build IDENTICAL features (no training-serving skew)."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import compute_features
from src.features.definitions import ALL_CANDIDATE_FEATURES, HISTORY_COLUMNS
from src.features.history_store import HistoryStore

NUMERIC = [c for c in ALL_CANDIDATE_FEATURES if c not in ("product_id", "store_id")]


@pytest.fixture(scope="module")
def store(raw):
    return HistoryStore(raw[HISTORY_COLUMNS])


@pytest.mark.parametrize("pid,sid,day", [("P01", "S1", "2024-12-10"), ("P04", "S2", "2024-03-05"), ("P08", "S3", "2024-11-01"),
                                         ("P05", "S2", "2023-06-20")])
def test_online_features_equal_offline_features(raw, store, pid, sid, day):
    offline = compute_features(raw[HISTORY_COLUMNS])
    off = offline[(offline.product_id == pid) & (offline.store_id == sid) & (offline.date == day)].iloc[0]
    r = raw[(raw.product_id == pid) & (raw.store_id == sid) & (raw.date == day)].iloc[0]
    ft = None if pd.isna(r.forecast_temp) else r.forecast_temp
    on = store.features_for(pid, sid, day, r.price, r.discount, int(r.promotion), int(r.holiday), ft).iloc[0]
    for c in NUMERIC:
        if pd.isna(off[c]):
            assert pd.isna(on[c]), c
        else:
            assert on[c] == pytest.approx(off[c], rel=1e-9, abs=1e-9), c


def test_store_never_uses_the_forecast_day_or_later(raw, store):
    day = pd.Timestamp("2024-12-10")
    r = raw[(raw.product_id == "P01") & (raw.store_id == "S1") & (raw.date == day)].iloc[0]
    base = store.features_for("P01", "S1", day, r.price, r.discount, int(r.promotion), int(r.holiday), r.forecast_temp)
    tampered = raw[HISTORY_COLUMNS].copy()
    m = (tampered.date >= day) & (tampered.product_id == "P01") & (tampered.store_id == "S1")
    tampered.loc[m, "sales"] = 99999.0                                           # corrupt today and the future
    other = HistoryStore(tampered).features_for("P01", "S1", day, r.price, r.discount, int(r.promotion), int(r.holiday), r.forecast_temp)
    pd.testing.assert_frame_equal(base[NUMERIC], other[NUMERIC])


def test_unknown_series_is_a_value_error(store):
    with pytest.raises(ValueError, match="unknown series"):
        store.features_for("P99", "S1", "2024-12-10", 10.0)


def test_stale_history_is_flagged_and_then_refused(raw):
    h = raw[HISTORY_COLUMNS]
    s = HistoryStore(h[h.date <= "2024-11-01"])
    assert s.staleness_days("P01", "S1", "2024-11-02") == 0
    assert s.staleness_days("P01", "S1", "2024-11-05") == 3
    s.features_for("P01", "S1", "2024-11-05", 45.0)                              # 3 days stale: still answers
    with pytest.raises(ValueError, match="behind"):
        s.features_for("P01", "S1", "2024-11-08", 45.0)                          # 6 days stale: refuses


def test_duplicated_history_refuses_to_load(raw):
    h = raw[HISTORY_COLUMNS]
    with pytest.raises(ValueError, match="duplicated"):
        HistoryStore(pd.concat([h, h.tail(5)]))
