"""The synthetic generator: schema, determinism and the real-world problems we promised to inject (docs/DATASET.md)."""
import numpy as np
import pandas as pd

from src.data import make_dataset
from src.data.make_dataset import generate, stress_level


def test_tables_and_columns(raw):
    assert set(raw) == {"customers", "transactions", "loan_status", "defaults"}
    assert list(raw["customers"].columns) == ["customer_id", "signup_date", "segment", "loan_amount", "loan_start_date", "credit_limit"]
    assert list(raw["transactions"].columns) == ["customer_id", "txn_date", "amount", "txn_type", "balance_after"]
    assert {"days_past_due", "missed_payment", "available_at", "collections_status", "recovery_amount"} <= set(raw["loan_status"].columns)
    assert len(raw["customers"]) == 1500 and raw["customers"]["customer_id"].is_unique
    assert len(raw["transactions"]) < 400_000                       # brief: keep the volume manageable


def test_transaction_types_and_amounts(raw):
    tx = raw["transactions"]
    assert set(tx["txn_type"]) == {"salary_credit", "loan_payment", "debit", "credit_card_spend", "atm", "other_credit"}
    assert (tx["amount"] > 0).all()
    assert tx.groupby("customer_id")["txn_date"].apply(lambda s: s.is_monotonic_increasing).all()      # sorted per customer


def test_loan_status_publication_lag(raw):
    ls = raw["loan_status"]
    assert ((ls["available_at"] - ls["month"]).dt.days == 5).all()      # month M is published on day 5 of M+1


def test_default_means_90_days_past_due(raw):
    ls, de = raw["loan_status"], raw["defaults"].dropna()
    row = ls.merge(de, left_on=["customer_id", "month"], right_on=["customer_id", "default_date"])
    assert len(row) == len(de) and (row["days_past_due"] >= 90).all()


def test_post_default_footprints_exist_for_the_leakage_lesson(raw):
    """After default: collections status + recoveries appear and the balance collapses (this is what leaks if a feature looks past as_of)."""
    ls, de, tx = raw["loan_status"], raw["defaults"].dropna(), raw["transactions"]
    after = ls.merge(de, on="customer_id")
    after = after[after["month"] > after["default_date"]]
    assert (after["collections_status"] == "in_collections").mean() > 0.9 and after["recovery_amount"].sum() > 0
    never = ls[~ls["customer_id"].isin(de["customer_id"])]
    assert (never["collections_status"] == "in_collections").sum() == 0 and never["recovery_amount"].sum() == 0
    t = tx.merge(de, on="customer_id")
    before = t[(t["txn_date"] > t["default_date"] - pd.Timedelta(days=120)) & (t["txn_date"] <= t["default_date"] - pd.Timedelta(days=60))]["balance_after"].median()
    later = t[(t["txn_date"] > t["default_date"] + pd.Timedelta(days=30)) & (t["txn_date"] <= t["default_date"] + pd.Timedelta(days=120))]["balance_after"].median()
    assert later < 0.4 * before


def test_injected_problems(raw):
    cu, tx = raw["customers"], raw["transactions"]
    assert 0.10 < cu["credit_limit"].isna().mean() < 0.25                       # no credit card -> missing limit
    oc = tx[tx["txn_type"] == "other_credit"]["amount"]
    assert oc.max() > 50 * oc.median()                                           # outlier windfalls
    assert set(cu["segment"]) == {"mass", "affluent", "small_business"}         # sensitive / proxy attribute


def test_stress_level_ramp():
    assert list(np.round(stress_level([31, 32, 33, 34, 50], 31), 2)) == [0.0, 0.33, 0.67, 1.0, 1.0]


def test_generator_is_deterministic_and_seed_sensitive(monkeypatch):
    small = {"data": {**make_dataset.load_params()["data"], "n_customers": 120}}
    monkeypatch.setattr(make_dataset, "load_params", lambda: small)
    a, b, c = generate(), generate(), generate(seed=1)
    pd.testing.assert_frame_equal(a["transactions"], b["transactions"])
    assert not a["transactions"].equals(c["transactions"])
