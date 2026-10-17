"""Feature tests: the golden rule (only data BEFORE the forecast day) and the worked examples from the docstrings."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import check_history, compute_features
from src.features.definitions import ALL_CANDIDATE_FEATURES
from src.features.manual_examples import toy_frame

SALES = [10, 12, 11, 13, 20, 22, 21, 14]            # the docstring example: days 1..8


def one_series(sales, **kw):
    return toy_frame(sales, **kw)


def test_lag_and_rolling_use_only_the_past():
    f = compute_features(one_series(SALES))
    day8 = f.iloc[7]
    assert day8["lag_1"] == 21 and day8["lag_7"] == 10
    assert day8["rolling_mean_7"] == pytest.approx(np.mean([10, 12, 11, 13, 20, 22, 21]))      # 15.57 (days 1..7)
    assert day8["rolling_mean_7"] != pytest.approx(np.mean([12, 11, 13, 20, 22, 21, 14]))      # 16.14 would include day 8 = leak


def test_changing_todays_sales_never_changes_todays_features():
    base = compute_features(one_series(SALES))
    changed = compute_features(one_series(SALES[:-1] + [999]))
    cols = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]
    pd.testing.assert_frame_equal(base.iloc[[7]][cols], changed.iloc[[7]][cols])


def test_features_are_computed_inside_each_series():
    a, b = one_series([100] * 10), one_series([5] * 10)
    b["product_id"] = "P02"
    f = compute_features(pd.concat([a, b], ignore_index=True))
    assert f[f.product_id == "P02"]["lag_1"].dropna().eq(5).all()          # never 100 (no leakage between series)


def test_missing_day_is_filled_with_recent_average_not_misaligned():
    f = compute_features(one_series([10, 12, 11, 13, 20, 22, 21, 14], drop_day=5))
    assert len(f) == 8                                                       # calendar completed: the gap is a row, not a hole
    assert np.isnan(f.iloc[4]["sales"])                                      # day 5 is the missing row
    # day 6 asks for "yesterday" (= missing day 5): it falls back to its own recent average instead of silently using day 4
    assert f.iloc[5]["lag_1"] == pytest.approx(f.iloc[5]["rolling_mean_7"])
    assert f.iloc[5]["lag_1"] != 13                                          # 13 would be the misaligned value (day 4)


def test_duplicated_dates_are_refused():
    df = one_series(SALES)
    with pytest.raises(ValueError, match="duplicated"):
        check_history(pd.concat([df, df.tail(2)]))
    with pytest.raises(ValueError):
        compute_features(pd.concat([df, df.tail(2)]))


def test_cyclical_encoding_examples():
    f = compute_features(one_series(SALES, start="2025-01-06"))              # a Monday
    r = f.iloc[0]
    assert (r["dow_sin"], r["dow_cos"]) == pytest.approx((0.0, 1.0))
    sunday = f.iloc[6]
    assert (sunday["dow_sin"], sunday["dow_cos"]) == pytest.approx((-0.7818, 0.6235), abs=1e-3)
    # Sunday and Monday are neighbours on the circle: distance of one step, not six
    d = np.hypot(sunday["dow_sin"] - r["dow_sin"], sunday["dow_cos"] - r["dow_cos"])
    assert d < 1.0


def test_price_features():
    df = one_series(SALES)
    df.loc[1, ["price", "discount"]] = [38.0, 2.0]
    df.loc[0, ["price", "discount"]] = [40.0, 0.0]
    f = compute_features(df)
    assert f.iloc[1]["price_change"] == pytest.approx(38 / 40 - 1)           # -0.05
    assert f.iloc[1]["discount_pct"] == pytest.approx(2 / 40)                # 0.05


def test_all_candidate_features_exist_and_are_finite_where_required(train_frame):
    for c in ALL_CANDIDATE_FEATURES:
        assert c in train_frame.columns
    level = train_frame[["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]]
    assert not level.isna().any(axis=None)
    assert train_frame["forecast_temp"].isna().any()                         # real missingness is kept (pipeline imputes it)


def test_warmup_rows_are_dropped(train_frame):
    assert train_frame.groupby(["product_id", "store_id"])["date"].min().min() >= pd.Timestamp("2023-01-31")
    assert train_frame["sales"].notna().all()


def test_trend_index_is_days_since_origin():
    f = compute_features(one_series(SALES, start="2023-01-11"))
    assert f.iloc[0]["trend_index"] == 10
