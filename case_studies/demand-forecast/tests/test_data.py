"""Data tests: the generator is deterministic and really contains the problems the course talks about."""
import numpy as np
import pandas as pd

from src.data import make_dataset as MD
from src.features.definitions import BATCH_COLUMNS, KEYS, RAW_COLUMNS


def test_shape_and_columns(raw):
    assert list(raw.columns) == RAW_COLUMNS
    assert raw.groupby(KEYS).ngroups == 24                       # 8 products x 3 stores
    assert raw["date"].nunique() == 731                          # 2023-01-01 .. 2024-12-31
    assert len(raw) == 24 * 731


def test_no_duplicate_rows(raw):
    assert not raw.duplicated(["date", *KEYS]).any()


def test_generator_is_deterministic(raw):
    again = MD.generate_all()["train"]
    pd.testing.assert_frame_equal(raw, again)


def test_injected_problems_exist(raw):
    assert raw["sales"].isna().any()                             # POS outage days (missing completely at random)
    assert raw["forecast_temp"].isna().any()                     # weather-service outage
    assert (raw["sales"] == 0).any()                             # stock-out zeros (breaks MAPE)
    p99 = raw["sales"].quantile(0.99)
    assert raw["sales"].max() > 2 * p99                          # bulk-order outliers
    assert (raw["promotion"] == 1).any() and ((raw["promotion"] == 1) & (raw["discount"] == 0)).any()   # display-only promotions


def test_actual_temperature_differs_from_forecast(raw):
    d = (raw["temperature"] - raw["forecast_temp"]).abs().dropna()
    assert d.mean() > 0.5                                        # a forecast is never the truth


def test_non_negative_counts_and_prices(raw):
    assert (raw["sales"].dropna() >= 0).all()
    assert (raw["price"] > 0).all() and (raw["discount"] >= 0).all()


def test_trend_shift_between_training_and_production(datasets):
    """Teaching table: average daily sales per series - training ~100..130 scale, drifted production ~150..210 scale."""
    tr = datasets["train"].groupby("date")["sales"].mean()
    dr = datasets["drifted"].groupby("date")["sales"].mean()
    last_train_month = tr[tr.index >= "2024-12-01"].mean()
    first_prod_week, last_prod_week = dr.iloc[:7].mean(), dr.iloc[-7:].mean()
    assert first_prod_week > 1.08 * last_train_month
    assert last_prod_week > 1.30 * last_train_month
    assert dr.iloc[-14:].mean() > dr.iloc[:14].mean() * 1.2     # the level keeps climbing


def test_normal_batch_continues_the_training_trend(datasets):
    tr = datasets["train"].groupby("date")["sales"].mean()
    nm = datasets["normal"].groupby("date")["sales"].mean()
    assert abs(nm.mean() / tr[tr.index >= "2024-12-01"].mean() - 1) < 0.25   # business as usual (+ seasonal effects)


def test_batch_layout_has_target_and_request_fields(datasets):
    b = MD.to_batch(datasets["drifted"])
    assert list(b.columns) == BATCH_COLUMNS
    assert "sales" in b.columns and "forecast_date" in b.columns
    assert b["forecast_date"].min() == pd.Timestamp("2025-01-01")


def test_price_shock_only_in_drifted_batch(datasets):
    a, b = datasets["normal"], datasets["drifted"]
    pa, pb = a[a.product_id == "P04"]["price"].median(), b[b.product_id == "P04"]["price"].median()
    assert pb > 1.05 * pa
