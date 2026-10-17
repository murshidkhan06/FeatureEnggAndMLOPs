"""Feature maths: hand calculation == code, derived features, target label, split."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (_slope_6, add_derived_features, add_target, assign_split, build_snapshot_features, month_end_shift)
from src.features.definitions import BASE_FEATURES, DERIVED_FEATURES, HORIZON_MONTHS, MODEL_FEATURES
from src.features.manual_examples import AS_OF, tiny_customer
from tests.conftest import needs_snapshots


@pytest.fixture(scope="module")
def tiny():
    t = tiny_customer()
    return build_snapshot_features(t["customers"], t["transactions"], t["loan_status"], AS_OF).iloc[0]


@pytest.mark.parametrize("name,expected", [
    ("salary_credit_frequency", 5 / 6), ("loan_payment_ratio", 5 / 6), ("missed_payment_count", 1), ("credit_utilization", 0.2),
    ("income_stability", 0.0707), ("recent_transaction_count", 4), ("average_balance", 112_000.0), ("balance_trend", -5_000.0), ("account_age", 36)])
def test_hand_calculation_equals_code(tiny, name, expected):
    assert tiny[name] == pytest.approx(expected, abs=1e-4)


def test_unpublished_loan_status_row_is_ignored():
    """The March row (missed=1) is published on 2023-04-05, so on 2023-03-31 the count stays 1 (a naive sum would give 2)."""
    t = tiny_customer()
    assert t["loan_status"]["missed_payment"].sum() == 2
    assert build_snapshot_features(t["customers"], t["transactions"], t["loan_status"], AS_OF).iloc[0]["missed_payment_count"] == 1


def test_balance_trend_three_point_example():
    assert _slope_6(np.array([[100.0, 80.0, 60.0]]))[0] == pytest.approx(-20.0)
    assert _slope_6(np.array([[1.0, 2, 3, 4, 5, 6]]))[0] == pytest.approx(1.0)


def test_derived_features():
    d = add_derived_features(pd.DataFrame({"credit_utilization": [0.5, np.nan], "missed_payment_count": [2, 1], "balance_trend": [-500.0, 90_000.0],
                                           "average_balance": [999.0, -10.0]}))
    assert d["util_x_missed"].iloc[0] == pytest.approx(1.0) and np.isnan(d["util_x_missed"].iloc[1])      # NaN passes through (imputed later)
    assert d["balance_trend_norm"].iloc[0] == pytest.approx(-0.5)
    assert d["balance_trend_norm"].iloc[1] == 1.0                                                          # clipped to [-1, 1]
    assert d["log_average_balance"].iloc[0] == pytest.approx(np.log1p(999.0)) and d["log_average_balance"].iloc[1] == 0.0   # negative balance -> 0


def test_feature_lists_are_consistent():
    assert MODEL_FEATURES == BASE_FEATURES + DERIVED_FEATURES and len(set(MODEL_FEATURES)) == 12


def test_target_window_boundaries():
    snap = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5], "as_of_date": [pd.Timestamp("2023-03-31")] * 5})
    de = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5], "default_date": pd.to_datetime(["2023-04-30", "2023-09-30", "2023-10-31", "2023-03-31", None])})
    # in (as_of, as_of + 6 months] -> 1 ; exactly 6 months counts, 7 months does not, 'already defaulted' / never -> 0
    assert add_target(snap, de)["default_next_6m"].tolist() == [1, 1, 0, 0, 0]


def test_month_end_shift():
    assert month_end_shift("2023-03-31", -6) == pd.Timestamp("2022-09-30") and month_end_shift("2023-03-31", HORIZON_MONTHS) == pd.Timestamp("2023-09-30")


def test_as_of_must_be_month_end(raw):
    with pytest.raises(ValueError):
        build_snapshot_features(raw["customers"], raw["transactions"], raw["loan_status"], "2023-03-15")


def test_customers_already_in_default_are_not_scored(raw):
    de = raw["defaults"].dropna()
    as_of = pd.Timestamp("2023-03-31")
    snap = build_snapshot_features(raw["customers"], raw["transactions"], raw["loan_status"], as_of, raw["defaults"])
    already = set(de.loc[de["default_date"] <= as_of, "customer_id"])
    assert len(already) > 0 and not (already & set(snap["customer_id"]))
    assert snap["customer_id"].is_unique and snap["as_of_date"].eq(as_of).all()


def test_time_split_is_ordered_with_purge_gap():
    from src.common.config import load_params
    sp = load_params()["splits"]
    dates = pd.Series(pd.date_range("2021-06-30", "2024-12-31", freq="ME"))
    s = assign_split(dates, sp)
    last = {k: dates[s == k].max() for k in ("train", "val", "test")}
    first = {k: dates[s == k].min() for k in ("val", "test", "production")}
    assert last["train"] < first["val"] and last["val"] < first["test"] and last["test"] < first["production"]
    gap_months = (first["val"].year - last["train"].year) * 12 + first["val"].month - last["train"].month
    assert gap_months >= HORIZON_MONTHS                       # labels of train rows are not yet known when validation starts
    assert set(s) == {"train", "purge", "val", "test", "production"}


@needs_snapshots
def test_snapshot_file_matches_definitions(snapshots):
    assert set(MODEL_FEATURES + ["customer_id", "as_of_date", "default_next_6m", "split"]) <= set(snapshots.columns)
    assert snapshots["default_next_6m"].isin([0, 1]).all() and 0.02 < snapshots["default_next_6m"].mean() < 0.10
    assert not snapshots.duplicated(["customer_id", "as_of_date"]).any()
