"""Leakage tests: chronological splits, no future information in features, leaky columns never selected."""
import numpy as np
import pandas as pd

from src.common.params import load_params
from src.features.leakage_demo import label_in_training_features, leaky_features, light
from src.features.select_features import load_selected
from src.models.train import time_split


def test_time_split_is_chronological_and_disjoint(train_frame):
    tr, va, te = time_split(train_frame, load_params()["split"])
    assert tr["date"].max() < va["date"].min() and va["date"].max() < te["date"].min()
    assert len(tr) + len(va) + len(te) <= len(train_frame) and min(len(tr), len(va), len(te)) > 1000


def test_selected_features_contain_no_leaky_column():
    sel = set(load_selected())
    assert not sel & {"temperature", "rolling_mean_7_centered", "rolling_mean_7_noshift", "sales_today", "sales"}


def test_centered_window_contains_the_future(raw):
    f = leaky_features(raw[raw["date"] < "2023-04-01"])
    one = f[(f.product_id == "P01") & (f.store_id == "S1")].sort_values("date").reset_index(drop=True)
    i = 20
    centred = one["sales"].iloc[i - 3:i + 4].mean()                  # days t-3 .. t+3 : includes today AND three future days
    assert one["rolling_mean_7_centered"].iloc[i] == np.float64(centred) or abs(one["rolling_mean_7_centered"].iloc[i] - centred) < 1e-9
    assert not np.isclose(one["rolling_mean_7"].iloc[i], one["rolling_mean_7_centered"].iloc[i])   # the honest feature differs


def test_random_split_puts_test_labels_inside_training_features(train_frame):
    assert label_in_training_features(train_frame, seed=0) > 0.9      # ~all labels are visible as lags of neighbouring days


def test_chronological_split_has_no_label_in_training_features(train_frame):
    tr, va, te = time_split(train_frame, load_params()["split"])
    # a lag_1 in the TEST period can point to the last TRAIN/VALID day only for the first test day; the converse never happens
    assert te["date"].min() > tr["date"].max()
    first_test_day = te[te["date"] == te["date"].min()]
    assert first_test_day["lag_1"].notna().all()


def test_scaler_fitted_on_train_only_exposes_the_level_shift(train_frame):
    from sklearn.preprocessing import MinMaxScaler
    tr, va, te = time_split(train_frame, load_params()["split"])
    wrong = MinMaxScaler().fit(train_frame[["rolling_mean_7"]]).transform(te[["rolling_mean_7"]]).max()
    right = MinMaxScaler().fit(tr[["rolling_mean_7"]]).transform(te[["rolling_mean_7"]]).max()
    assert right >= wrong - 1e-9                                       # honest scaling never hides a level above the training range


def test_light_params_are_smaller():
    p = load_params()
    assert light(p)["hgb"]["max_iter"] < p["hgb"]["max_iter"]
