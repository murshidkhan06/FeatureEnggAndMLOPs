"""EARLY (PROXY) EVIDENCE while the real labels are still months away: the LABEL LAG PROBLEM.

THE PROBLEM
  The label "defaulted within 6 months of as_of" is only known 6 months after as_of.  A model scored in January 2024 cannot be
  graded before July 2024.  If we wait for labels, a stressed portfolio runs for half a year with a model nobody has checked.

THREE KINDS OF EARLIER EVIDENCE (cheap -> strong)
  1. input drift (KS test)                 available immediately, but says nothing about default itself
  2. score shift                           mean predicted PD and the share of customers above the contact threshold
                                           (a model that suddenly flags 25 % of customers instead of 10 % overloads collections)
  3. early-delinquency rate (PSEUDO-LABEL) share of customers with >= 1 missed instalment in the 3 months AFTER as_of.
                                           Known after ~3 months (+ the 5-day publication lag) instead of 6.  Not the true label
                                           (a missed instalment is not a default) but moves in the same direction.

TINY EXAMPLE (pseudo-label)
  1,000 customers scored on 2024-01-31.  By 2024-05-05 the loan_status for Feb, Mar, Apr is published.
  60 customers missed >= 1 instalment -> early-delinquency rate 6 %.  Normal months show 3 % -> ratio 2.0 (>= 1.5: evidence).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import month_end_shift
from src.features.definitions import HORIZON_MONTHS


def label_available_on(as_of) -> pd.Timestamp:
    """The first month-end on which the 6-month label of as_of is fully known.  2024-01-31 -> 2024-07-31."""
    return month_end_shift(pd.Timestamp(as_of), HORIZON_MONTHS)


def labels_mature(as_of, today) -> bool:
    return label_available_on(as_of) <= pd.Timestamp(today)


def early_delinquency_rate(loan_status: pd.DataFrame, customer_ids, as_of, today, months: int = 3) -> float | None:
    """Share of customers with >= 1 missed instalment in the `months` months after as_of (only rows PUBLISHED by `today`).

    Returns None if the 3 months are not fully published yet (the pseudo-label is not ready)."""
    as_of, today = pd.Timestamp(as_of), pd.Timestamp(today)
    end = month_end_shift(as_of, months)
    if end + pd.Timedelta(days=5) > today:
        return None
    w = loan_status[(loan_status["month"] > as_of) & (loan_status["month"] <= end) & (loan_status["available_at"] <= today)
                    & loan_status["customer_id"].isin(customer_ids)]
    missed = w.groupby("customer_id")["missed_payment"].sum()
    return float((missed.reindex(pd.Index(customer_ids)).fillna(0) >= 1).mean())


def score_shift(champion: dict, prod_scores: np.ndarray, ref_mean: float, top_pct: float = 0.10) -> dict:
    """Mean predicted PD vs reference, and the share of customers above the FIXED contact threshold (capacity check)."""
    contact = float((prod_scores >= champion["threshold"]).mean())
    return {"mean_pd": round(float(np.mean(prod_scores)), 4), "ref_mean_pd": round(ref_mean, 4),
            "mean_pd_ratio": round(float(np.mean(prod_scores)) / ref_mean, 2),
            "contact_rate": round(contact, 4), "planned_contact_rate": top_pct, "contact_rate_ratio": round(contact / top_pct, 2)}
