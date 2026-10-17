"""HAND CALCULATION vs CODE for the main features, on ONE tiny customer you can verify with a pencil.

Why: before trusting a feature pipeline that runs on 300,000 transactions, prove on a toy customer that the numbers equal
what a human gets by hand.  Every feature below is computed by the REAL function  build_snapshot_features()  and compared with
a value calculated by hand (the comments show the arithmetic).

THE CUSTOMER  (as_of = 2023-03-31, window = Oct 2022 .. Mar 2023, signup 2020-03-15, credit limit 30,000, loan since 2022-01-15)
  month   salary    loan payment   debit     card spend   month-end balance
  Oct     100,000   20,000         85,000    6,000        95,000        (opening balance 100,000)
  Nov     110,000   20,000         95,000    6,000        90,000
  Dec      90,000   20,000         75,000    6,000        85,000
  Jan        -         -            5,000    6,000        80,000        <- salary and loan payment MISSING (missed instalment)
  Feb     100,000   20,000         85,000    6,000        75,000
  Mar     100,000   20,000         85,000    6,000        70,000
  loan_status: Jan missed=1.  March row (missed=1) is published on 2023-04-05 -> NOT available on 2023-03-31 -> must be ignored.

Run:  python -m src.features.manual_examples
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.facts import save_fact
from src.features.build_features import _slope_6, add_derived_features, add_target, build_snapshot_features

AS_OF = pd.Timestamp("2023-03-31")


def tiny_customer() -> dict:
    """Build the four raw tables for the toy customer (customer_id = 1)."""
    rows, bal = [], 100_000.0
    salary = {10: 100_000, 11: 110_000, 12: 90_000, 2: 100_000, 3: 100_000}          # January: none
    debit = {10: 85_000, 11: 95_000, 12: 75_000, 1: 5_000, 2: 85_000, 3: 85_000}
    for y, m in ((2022, 10), (2022, 11), (2022, 12), (2023, 1), (2023, 2), (2023, 3)):
        def add(day, typ, amt, sign):
            nonlocal bal
            bal += sign * amt
            rows.append((1, pd.Timestamp(y, m, day), float(amt), typ, bal))
        if m in salary:
            add(2, "salary_credit", salary[m], +1)
        if m != 1:
            add(8, "loan_payment", 20_000, -1)
        add(15, "debit", debit[m], -1)
        add(20, "credit_card_spend", 6_000, 0)
    tx = pd.DataFrame(rows, columns=["customer_id", "txn_date", "amount", "txn_type", "balance_after"])
    customers = pd.DataFrame({"customer_id": [1], "signup_date": [pd.Timestamp("2020-03-15")], "segment": ["mass"], "loan_amount": [500_000.0],
                              "loan_start_date": [pd.Timestamp("2022-01-15")], "credit_limit": [30_000.0]})
    months = pd.date_range("2022-09-30", "2023-03-31", freq="ME")
    ls = pd.DataFrame({"customer_id": 1, "month": months, "days_past_due": 0, "missed_payment": [1 if m == pd.Timestamp("2023-01-31") or m == pd.Timestamp("2023-03-31") else 0 for m in months],
                       "available_at": months + pd.Timedelta(days=5), "collections_status": "none", "recovery_amount": 0.0})
    return {"customers": customers, "transactions": tx, "loan_status": ls}


def main() -> None:
    t = tiny_customer()
    f = build_snapshot_features(t["customers"], t["transactions"], t["loan_status"], AS_OF).iloc[0]
    d = add_derived_features(pd.DataFrame([f[["average_balance", "balance_trend", "credit_utilization", "missed_payment_count"]]]).astype(float)).iloc[0]
    # ---- hand calculations -------------------------------------------------------------------------------------------
    manual = {
        # salary credits in the window: Oct, Nov, Dec, Feb, Mar = 5 ;  5 / 6 months
        "salary_credit_frequency": 5 / 6,
        # loan payments made = 5 (January missed) ; due = 6 (loan started 2022-01, every window month) ; 5 / 6
        "loan_payment_ratio": 5 / 6,
        # missed_payment in the PUBLISHED months Sep..Feb: only January = 1.  The March row (available 2023-04-05) is NOT usable.
        "missed_payment_count": 1,
        # card spend 6,000 in each of 6 months -> total 36,000 ; per month 6,000 ; / limit 30,000 = 0.2
        "credit_utilization": 0.2,
        # salaries 100,110,90,100,100 (thousand): mean 100 ; deviations 0,+10,-10,0,0 ; squares 0,100,100,0,0 = 200 ;
        # sample variance = 200 / (5 - 1) = 50 ; std = 7.0711 ; CV = 7.0711 / 100 = 0.0707
        "income_stability": float(np.sqrt(50) / 100),     # code rounds to 4 decimals -> compare with a 1e-4 tolerance
        # transactions after 2023-03-01: salary d2, loan d8, debit d15, card d20 = 4
        "recent_transaction_count": 4,
        # balance_after of the 10 transactions since 2023-01-01 (Jan: 80,80 ; Feb: 180,160,75,75 ; Mar: 175,155,70,70) -> sum 1,120 / 10 = 112 (thousand)
        "average_balance": 112_000.0,
        # month-end balances 95,90,85,80,75,70 (thousand): a straight line down by 5 per month -> slope = -5,000
        "balance_trend": -5_000.0,
        # whole months from 2020-03-15 to 2023-03-31 = (2023-2020)*12 + (3-3) = 36
        "account_age": 36,
        # derived: 0.2 * 1 ; -5000 / (112000 + 1) ; log(1 + 112000)
        "util_x_missed": 0.2,
        "balance_trend_norm": -5_000 / 112_001,
        "log_average_balance": float(np.log1p(112_000)),
    }
    code = {k: (float(f[k]) if k in f.index else float(d[k]) if k in d.index else float("nan")) for k in manual}
    d_all = add_derived_features(pd.DataFrame([f.drop(["customer_id", "as_of_date"])]).astype(float)).iloc[0]
    code.update({k: float(d_all[k]) for k in ("util_x_missed", "balance_trend_norm", "log_average_balance")})
    print(f"Toy customer, as_of = {AS_OF.date()}  (see the module docstring for the monthly table)\n")
    print(f"{'feature':<26}{'by hand':>14}{'code':>14}  match")
    ok_all, table = True, {}
    for k, hv in manual.items():
        ok = bool(np.isclose(hv, code[k], rtol=1e-4, atol=1e-4))
        ok_all &= ok
        table[k] = dict(by_hand=round(float(hv), 5), code=round(code[k], 5), match=ok)
        print(f"{k:<26}{hv:>14.4f}{code[k]:>14.4f}  {'OK' if ok else 'MISMATCH'}")
    print(f"\nALL FEATURES MATCH THE HAND CALCULATION: {ok_all}")

    print("\nBALANCE TREND on three points (the textbook example):  balances [100, 80, 60] at x = 0, 1, 2")
    y = np.array([[100.0, 80.0, 60.0]])
    print("   x_mean = 1, y_mean = 80 ; slope = sum((x-1)(y-80)) / sum((x-1)^2) = ((-1)(+20) + (0)(0) + (+1)(-20)) / (1 + 0 + 1) = -40 / 2 = -20")
    print(f"   code (_slope_6 works for any number of columns): {_slope_6(y)[0]:.1f}   numpy.polyfit cross-check: {np.polyfit([0, 1, 2], y[0], 1)[0]:.1f}")

    print("\nAVAILABILITY RULE on the same customer: what if the unpublished March loan_status row (missed=1) were used by mistake?")
    wrong = int(t["loan_status"]["missed_payment"].sum())
    print(f"   correct missed_payment_count (published rows only) = {int(f['missed_payment_count'])};  naive sum over ALL rows = {wrong}  <- one future row inflates the feature")

    print("\nTARGET LABEL on the same logic:  default_next_6m = 1 if default_date in (as_of, as_of + 6 months]")
    snap = pd.DataFrame({"customer_id": [1, 2, 3], "as_of_date": [AS_OF] * 3})
    de = pd.DataFrame({"customer_id": [1, 2, 3], "default_date": [pd.Timestamp("2023-08-31"), pd.Timestamp("2023-10-31"), pd.NaT]})
    lab = add_target(snap, de)
    for (_, r), dd in zip(lab.iterrows(), de["default_date"]):
        print(f"   customer {int(r['customer_id'])}: default_date {dd.date() if pd.notna(dd) else 'never'}  -> label {int(r['default_next_6m'])}")
    save_fact("feature_manual_examples", dict(features=table, all_match=ok_all, slope_example=dict(y=[100, 80, 60], slope=-20.0),
                                              naive_missed_with_future_row=wrong, correct_missed=int(f["missed_payment_count"])))


if __name__ == "__main__":
    main()
