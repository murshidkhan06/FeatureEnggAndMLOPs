"""The FUTURE-PERTURBATION TEST: the strongest practical test that a feature pipeline does not peek into the future.

IDEA
  At prediction time the future does not exist yet.  So if we CORRUPT everything that is not yet available at
  as_of_date (future transactions, loan_status rows not yet published) and rebuild the features, a leakage-safe
  pipeline must return EXACTLY the same numbers.  If any number changes, that feature looked into the future.

TINY EXAMPLE
  as_of = 2023-03-31.  A future transaction (2023-05-10, balance 90,000) is replaced by balance 123,456,789.
    safe feature   "mean balance over the last 90 days up to 2023-03-31" -> unchanged   -> test PASSES
    leaky feature  "mean balance over the 90 days up to 2023-09-30"      -> changes     -> test FAILS
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def perturb_future(transactions: pd.DataFrame, loan_status: pd.DataFrame, as_of, seed: int = 0):
    """Return corrupted copies of the raw tables where every row NOT YET AVAILABLE at as_of is scrambled."""
    as_of = pd.Timestamp(as_of)
    rng = np.random.default_rng(seed)
    tx, ls = transactions.copy(), loan_status.copy()
    f = (tx["txn_date"] > as_of).to_numpy()
    tx.loc[f, "amount"] = tx.loc[f, "amount"].to_numpy() * rng.uniform(0.1, 10, f.sum())
    tx.loc[f, "balance_after"] = rng.normal(0, 1e7, f.sum())
    tx.loc[f, "txn_type"] = rng.permutation(tx.loc[f, "txn_type"].to_numpy())
    g = (ls["available_at"] > as_of).to_numpy()
    ls.loc[g, "missed_payment"] = rng.integers(0, 2, g.sum())
    ls.loc[g, "days_past_due"] = rng.integers(0, 181, g.sum())
    ls.loc[g, "recovery_amount"] = rng.uniform(0, 1e5, g.sum())
    ls.loc[g, "collections_status"] = rng.choice(["none", "arrears", "in_collections"], g.sum())
    return tx, ls


def future_perturbation_test(builder: Callable[..., pd.DataFrame], raw: dict, as_of, seed: int = 0) -> dict:
    """Run `builder(customers, transactions, loan_status, as_of)` on clean and on future-corrupted data.

    Returns {'passed': bool, 'changed_columns': [...], 'n_future_tx_rows': int}."""
    as_of = pd.Timestamp(as_of)
    clean = builder(raw["customers"], raw["transactions"], raw["loan_status"], as_of)
    tx2, ls2 = perturb_future(raw["transactions"], raw["loan_status"], as_of, seed)
    dirty = builder(raw["customers"], tx2, ls2, as_of)
    changed = []
    same_rows = clean["customer_id"].tolist() == dirty["customer_id"].tolist()
    if same_rows:
        for c in clean.columns:
            a, b = clean[c], dirty[c]
            if pd.api.types.is_numeric_dtype(a):
                if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=0, atol=0):
                    changed.append(c)
            elif not a.equals(b):
                changed.append(c)
    return {"passed": same_rows and not changed, "changed_columns": changed if same_rows else ["<row set changed>"],
            "n_future_tx_rows": int((raw["transactions"]["txn_date"] > as_of).sum())}
