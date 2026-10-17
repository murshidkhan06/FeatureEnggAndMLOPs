"""Synthetic bank data generator (seeded, 100% generated, no downloads).

WHY this file exists
  Real bank data is confidential.  We simulate 1,500 retail customers (current account + personal loan) over 54
  months and write FOUR RAW TABLES, exactly like a data engineer would receive them from core-banking systems:

    customers.csv     customer_id, signup_date, segment, loan_amount, loan_start_date, credit_limit
    transactions.csv  customer_id, txn_date, amount, txn_type, balance_after
    loan_status.csv   customer_id, month, days_past_due, missed_payment, available_at,
                      collections_status, recovery_amount        (<- last two columns are AFTER-THE-FACT columns)
    defaults.csv      customer_id, default_date (empty = never defaulted)

  DEFAULT DEFINITION: 90+ days past due.  days_past_due = 30 x (number of consecutive missed instalments), so
  three missed instalments in a row = 90 dpd = default.  default_date = month-end of the third missed month.

HOW the story is simulated (month by month, vectorised over customers)
  1. Every customer has a hidden risk level z ~ N(0, 1).  Higher z -> higher default hazard AND slightly worse
     everyday behaviour (more card use, more irregular salary).  Analysts never see z; they see behaviour.
  2. Monthly default hazard  h = base * exp(0.75 z) * danger_zones * (1 + 2.0 * stress)       (stress = macro shock, see below)
     Worked example: base=0.0065, z=0, stress=0  ->  h = 0.65 % per month  ->  about 3.8 % over 6 months.
  3. A customer who WILL default in month D shows warning signs before D (this is what the model must find):
       type A (gradual,  ~76 % normally): balance falls, card utilisation rises, salary gets irregular, misses begin
       type B (job loss, ~12 % normally, ~50 % under stress): salary credits stop 4 months before D
       type C (sudden,   ~12 %): looks normal until the three forced missed instalments D-2, D-1, D
     Some customers who never default also have "false alarm" distress episodes (so the task is not trivial).
  4. In the default month D (day 26) the bank sweeps the account -> the BALANCE COLLAPSES; afterwards salary stops, card is blocked,
     loan_status shows growing days_past_due, collections_status = in_collections and recoveries appear.
     These post-default footprints are what makes the leakage lesson possible.
  5. MACRO STRESS starts in 2023-09 (month index 32) and is full from 2023-11: fewer salary credits, higher card
     utilisation, more irregular salary, higher hazard, and more job-loss type defaults  ->  data drift AND a rising
     default rate AND (because of type B) a partly changed relationship between features and default.

REAL-WORLD PROBLEMS INJECTED  (documented in docs/DATASET.md)
  * missing values: credit_limit is empty for customers without a credit card (MAR: depends on segment/income);
    income_stability becomes undefined when fewer than 2 salary credits exist (MNAR: the missingness IS the risk).
  * outliers: ~0.5 % of customers receive one huge one-off credit (property sale, inheritance).
  * class imbalance: ~5 % of snapshots are positives.
  * correlated features: missed_payment_count vs loan_payment_ratio, balance_trend vs average_balance ...
  * late-arriving data: loan_status is published 5 days after month-end (available_at).
  * leakage columns: collections_status, recovery_amount, days_past_due of the future months.
  * sensitive / proxy attribute: segment (small_business customers have irregular income by construction).

Run:  python -m src.data.make_dataset      (about 10 seconds)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import load_params
from src.common.paths import DATA_RAW

SEGMENTS = ["mass", "affluent", "small_business"]
LOAN_TERM_MONTHS = 84          # every loan is a 7-year loan
INTEREST_FACTOR = 1.35         # total repaid = loan_amount * 1.35 -> instalment = loan_amount * 1.35 / 84
LOAN_STATUS_LAG_DAYS = 5       # month M is published on day 5 of month M+1
NEVER = 10_000                 # sentinel month index meaning "never defaults"


def stress_level(t: int | np.ndarray, onset: int) -> float | np.ndarray:
    """Macro stress in [0, 1].  stress(t) = clip((t - onset) / 3, 0, 1).

    Example (onset=31): t=31 -> 0.0, t=32 -> 0.33, t=33 -> 0.67, t>=34 -> 1.0 (ramps up over three months)."""
    return np.clip((np.asarray(t, dtype=float) - onset) / 3.0, 0.0, 1.0)


def _month_index(dates: np.ndarray, first_month_end: pd.Timestamp) -> np.ndarray:
    """Month index of each date: 2021-01-xx -> 0, 2021-02-xx -> 1, ..."""
    d = pd.DatetimeIndex(dates)
    return (d.year - first_month_end.year) * 12 + (d.month - first_month_end.month)


def _make_customers(rng: np.random.Generator, n: int, first_me: pd.Timestamp) -> dict:
    """Draw static customer attributes + hidden behavioural traits (traits are never written to disk)."""
    seg = rng.choice(SEGMENTS, n, p=[0.60, 0.25, 0.15])
    mult = np.select([seg == "affluent", seg == "mass"], [2.0, 0.85], default=1.0)
    income = np.clip(rng.lognormal(np.log(42000), 0.40, n) * mult, 15000, 350000).round(-2)
    z = rng.normal(0, 1, n) + 0.3 * (seg == "small_business")           # hidden risk level

    is_new = rng.random(n) < 0.18                                          # 18 % signed up during the 54 months
    old_signup = np.datetime64("1998-01-01") + rng.integers(0, 8400, n).astype("timedelta64[D]")  # up to 2020-12
    new_signup = np.datetime64("2021-01-01") + rng.integers(0, 1095, n).astype("timedelta64[D]")  # 2021-01..2023-12
    signup = np.where(is_new, new_signup, old_signup).astype("datetime64[D]")
    s_idx = np.where(is_new, _month_index(signup, first_me) + 1, 0)       # first FULL month of activity

    dti = np.clip(0.22 + 0.05 * z + rng.normal(0, 0.04, n), 0.08, 0.42)    # instalment / income
    loan_amount = np.maximum(((income * dti) * LOAN_TERM_MONTHS / INTEREST_FACTOR).round(-3), 20000.0)
    instalment = loan_amount * INTEREST_FACTOR / LOAN_TERM_MONTHS
    old_loan = np.datetime64("2018-01-01") + rng.integers(0, 1095, n).astype("timedelta64[D]")
    loan_start = np.where(is_new, signup + rng.integers(0, 45, n).astype("timedelta64[D]"),
                          np.maximum(old_loan, signup)).astype("datetime64[D]")
    first_due = np.maximum(_month_index(loan_start, first_me) + 1, s_idx)  # first month an instalment is due

    p_card = np.select([seg == "affluent", seg == "mass"], [0.98, 0.78], default=0.88)
    has_card = rng.random(n) < p_card
    credit_limit = np.where(has_card, (income * rng.uniform(1.5, 5.0, n)).round(-3), np.nan)

    return dict(
        seg=seg, income=income, z=z, signup=signup, s_idx=s_idx, loan_amount=loan_amount, instalment=instalment,
        loan_start=loan_start, first_due=first_due, has_card=has_card, credit_limit=credit_limit,
        payday=rng.integers(1, 26, n), loan_day=rng.integers(5, 11, n),
        p_sal_base=0.985 - 0.03 * np.maximum(z, 0),
        sal_sigma=0.03 + 0.025 * np.maximum(z, 0) + 0.06 * (seg == "small_business"),
        util_base=np.clip(0.22 + 0.07 * z + rng.normal(0, 0.07, n), 0.02, 0.90),
        spend_base=np.clip(0.68 + 0.04 * z + rng.normal(0, 0.06, n), 0.35, 0.90),
        target_bal=income * rng.uniform(0.5, 3.0, n),
        windfall_month=np.where(rng.random(n) < 0.005, rng.integers(6, 40, n), NEVER),
    )


def _plan_defaults(rng: np.random.Generator, c: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decide WHO defaults and WHEN (planned default month D, -1 sentinel NEVER) and of which warning-sign type."""
    n, T = len(c["z"]), p["n_months"]
    D = np.full(n, NEVER)
    # threshold ("danger zone") effects that a straight line cannot capture but a tree can:
    #   heavy card users (habitual utilisation above 35 %) and customers with a thin cash buffer (< 0.8 x monthly income)
    danger_zones = (1 + 4.0 * (c["has_card"] & (c["util_base"] > 0.35))) * (1 + 0.8 * (c["target_bal"] < 0.8 * c["income"]))
    for t in range(T):
        s = float(stress_level(t, p["stress_onset_month"]))
        h = p["base_monthly_hazard"] * np.exp(0.75 * c["z"]) * danger_zones * (1 + 2.0 * s)
        ok = (D == NEVER) & (t >= c["first_due"] + 3)                     # needs 3 instalments to exist
        D[ok & (rng.random(n) < h)] = t
    s_at_D = stress_level(np.minimum(D, T - 1), p["stress_onset_month"])
    p_b = 0.12 + 0.38 * s_at_D                                            # job-loss share rises under stress
    u = rng.random(n)
    dtype = np.where(u < p_b, 2, np.where(u < p_b + 0.22, 3, 1))          # 1=gradual 2=job loss 3=sudden
    return D, dtype


def _distress_channels(rng: np.random.Generator, D: np.ndarray, dtype: np.ndarray, T: int):
    """Build three (customer x month) matrices in [0,1]: financial stress, salary stress, missed-payment stress."""
    n = len(D)
    r_fin, r_sal, r_miss = (np.zeros((n, T)) for _ in range(3))
    t = np.arange(T)
    iota = rng.uniform(0.25, 0.85, n)                                      # how strong the warning signs are
    for i in np.where(D < NEVER)[0]:
        d = D[i]
        ramp = np.clip((t - (d - 8)) / 7.0, 0, 1)
        ramp[t > d] = 0
        if dtype[i] == 1:                                                 # gradual deterioration
            r_fin[i] = r_sal[i] = r_miss[i] = iota[i] * ramp
        elif dtype[i] == 2:                                               # job loss: salary stops, rest looks OK
            r_sal[i] = np.where((t >= d - 4) & (t <= d), 1.0, np.where(t == d - 5, 0.5, 0.0))
            r_fin[i] = 0.25 * iota[i] * ramp
            r_miss[i] = 0.15 * ramp
        else:                                                             # sudden: only a short last-minute spike
            r_fin[i, max(d - 2, 0):d] = 0.6
    # false alarms: ~35 % of customers go through a temporary rough patch (even if they never default)
    has_ep = rng.random(n) < 0.35
    start, length = rng.integers(3, T - 8, n), rng.integers(4, 8, n)
    peak, salary_type = rng.uniform(0.3, 0.8, n), rng.random(n) < 0.3
    for i in np.where(has_ep)[0]:
        tri = np.clip(peak[i] * (1 - np.abs(t - (start[i] + length[i] / 2)) / (length[i] / 2)), 0, None)
        if salary_type[i]:
            r_sal[i] = np.maximum(r_sal[i], tri)
        else:
            r_fin[i] = np.maximum(r_fin[i], tri)
            r_miss[i] = np.maximum(r_miss[i], tri)
    return r_fin, r_sal, r_miss


def generate(seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Simulate everything and return the four raw tables as DataFrames."""
    p = load_params()["data"]
    rng = np.random.default_rng(p["seed"] if seed is None else seed)
    n, T = p["n_customers"], p["n_months"]
    first_me = pd.Timestamp(p["first_month_end"])
    month_ends = pd.date_range(first_me, periods=T, freq="ME")
    month_starts = (month_ends - pd.offsets.MonthBegin(1)).values.astype("datetime64[D]")
    c = _make_customers(rng, n, first_me)
    D, dtype = _plan_defaults(rng, c, p)
    r_fin, r_sal, r_miss = _distress_channels(rng, D, dtype, T)

    cid = np.arange(1, n + 1)
    income, inst = c["income"], c["instalment"]
    # start every account near its long-run level (avoids an artificial "everyone's balance is rising" warm-up trend)
    equilibrium = c["target_bal"] + income * (1.0 - c["spend_base"] - inst / income) / 0.12
    bal = np.maximum(equilibrium, 0.2 * income) * rng.uniform(0.85, 1.15, n)
    opening = bal.copy()
    streak = np.zeros(n, dtype=int)
    txn_parts: list[pd.DataFrame] = []
    ls_parts: list[pd.DataFrame] = []

    def add(mask, day, amount, kind, month_t):
        """Append one transaction per selected customer (vectorised)."""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        dates = month_starts[month_t] + (np.asarray(day)[idx] - 1).astype("timedelta64[D]")
        txn_parts.append(pd.DataFrame({"customer_id": cid[idx], "txn_date": dates,
                                       "amount": np.asarray(amount)[idx].round(2), "txn_type": kind}))

    for t in range(T):
        s = float(stress_level(t, p["stress_onset_month"]))
        active = t >= c["s_idx"]
        frozen = active & (t > D)                                         # after the default month
        live = active & ~frozen
        # ---- salary ---------------------------------------------------------------------------
        p_sal = np.clip(c["p_sal_base"] - 0.08 * s - 0.80 * r_sal[:, t], 0.02, 0.995)
        sal_paid = live & (rng.random(n) < p_sal)
        sigma = c["sal_sigma"] + 0.04 * s + 0.20 * r_sal[:, t]
        sal_amt = income * np.exp(rng.normal(0, 1, n) * sigma) * (1 - 0.20 * r_sal[:, t])
        add(sal_paid, np.clip(c["payday"] + rng.integers(-1, 2, n), 1, 28), sal_amt, "salary_credit", t)
        # ---- loan instalment (missed or paid) --------------------------------------------------
        due = live & (t >= c["first_due"])
        forced = due & (D < NEVER) & (t >= D - 2) & (t <= D)
        forbid = due & (D < NEVER) & (t == D - 3)
        p_miss = 0.012 + 0.012 * np.maximum(c["z"], 0) + 0.015 * s + 0.40 * r_miss[:, t]
        rand_miss = due & ~forbid & (streak < 2) & (rng.random(n) < p_miss)
        miss = forced | rand_miss
        paid_loan = due & ~miss
        add(paid_loan, c["loan_day"], inst, "loan_payment", t)
        # ---- everyday spending (cash-flow aware: spends less when the balance is low) ----------------
        want = income * c["spend_base"] * (1 + 0.35 * r_fin[:, t] + 0.03 * s) + 0.12 * (bal - c["target_bal"])
        want = np.maximum(want, 0.15 * income)
        room = np.maximum(bal + np.where(sal_paid, sal_amt, 0) - np.where(paid_loan, inst, 0) + 0.25 * income, 0)
        spend = np.where(live, np.minimum(want, room), 0.0)
        has_atm = live & (rng.random(n) < 0.55)
        atm_amt = np.where(has_atm, 0.12 * spend, 0.0)
        add(has_atm, rng.integers(1, 29, n), atm_amt.round(-2), "atm", t)
        debit_total = spend - np.where(has_atm, atm_amt.round(-2), 0)
        two = live & (rng.random(n) < 0.30)
        w = rng.uniform(0.3, 0.7, n)
        add(live, rng.integers(1, 29, n), np.where(two, debit_total * w, debit_total), "debit", t)
        add(two, rng.integers(1, 29, n), debit_total * (1 - w), "debit", t)
        # ---- credit card (memo only: purchases do not move the current-account balance) --------------
        card_live = live & c["has_card"]
        util = np.clip(c["util_base"] + 0.07 * s + 0.55 * r_fin[:, t] + rng.normal(0, 0.05, n), 0.0, 1.15)
        card_total = np.nan_to_num(util * c["credit_limit"])
        two_c = card_live & (rng.random(n) < 0.40)
        w2 = rng.uniform(0.3, 0.7, n)
        add(card_live, rng.integers(1, 29, n), np.where(two_c, card_total * w2, card_total), "credit_card_spend", t)
        add(two_c, rng.integers(1, 29, n), card_total * (1 - w2), "credit_card_spend", t)
        # ---- other credits (small gifts / refunds, plus rare huge windfalls = outliers) ---------------
        oc = live & (rng.random(n) < 0.18)
        oc_amt = income * rng.uniform(0.03, 0.30, n)
        windfall = live & (c["windfall_month"] == t)
        oc_amt = np.where(windfall, income * rng.uniform(10, 25, n), oc_amt)
        add(oc | windfall, rng.integers(1, 29, n), oc_amt, "other_credit", t)
        # ---- balance at the end of the month BEFORE any sweep ----------------------------------------
        bal_end = (bal + np.where(sal_paid, sal_amt, 0) + np.where(oc | windfall, oc_amt, 0)
                   - np.where(paid_loan, inst, 0) - spend)
        # ---- default month: the bank sets off the balance against the loan (ACCOUNT SWEEP), then (almost) nothing ----
        sweep = active & (t == D) & (bal_end > 0)
        add(sweep, np.full(n, 26), 0.92 * bal_end, "debit", t)
        trickle = frozen & (rng.random(n) < 0.30)
        add(trickle, rng.integers(1, 29, n), np.minimum(rng.uniform(500, 2500, n), np.maximum(bal, 0)).round(-2), "atm", t)
        bal = bal_end - np.where(sweep, 0.92 * bal_end, 0)           # running balance for the NEXT month's spending rule
        bal = np.where(active, bal, opening)
        # ---- loan_status rows (one per customer-month once an instalment is due) ----------------------
        streak = np.where(active & (t >= c["first_due"]), np.where(miss | frozen, streak + 1, 0), streak)
        in_status = active & (t >= c["first_due"])
        dpd = np.minimum(30 * streak, 180)
        rec = np.where(frozen & (t > D) & (rng.random(n) < 0.5), inst * rng.uniform(0.1, 0.6, n), 0.0).round(2)
        sel = np.where(in_status)[0]
        ls_parts.append(pd.DataFrame({
            "customer_id": cid[sel], "month": month_ends[t],
            "days_past_due": dpd[sel], "missed_payment": (miss | frozen)[sel].astype(int),
            "available_at": month_ends[t] + pd.Timedelta(days=LOAN_STATUS_LAG_DAYS),
            "collections_status": np.where(dpd[sel] >= 90, "in_collections", np.where(dpd[sel] > 0, "arrears", "none")),
            "recovery_amount": rec[sel]}))

    # ---- assemble tables ------------------------------------------------------------------------------------
    tx = pd.concat(txn_parts, ignore_index=True)
    tx["txn_date"] = pd.to_datetime(tx["txn_date"])
    order = {"salary_credit": 0, "other_credit": 1, "loan_payment": 2, "debit": 3, "atm": 4, "credit_card_spend": 5}
    tx["_o"] = tx["txn_type"].map(order)
    tx = tx.sort_values(["customer_id", "txn_date", "_o"], kind="stable").reset_index(drop=True)
    tx = tx[tx["amount"] > 0].reset_index(drop=True)                      # rounding can create 0-amount rows: drop them
    tx["_s"] = tx["amount"] * tx["txn_type"].map({"salary_credit": 1, "other_credit": 1, "loan_payment": -1,
                                                  "debit": -1, "atm": -1, "credit_card_spend": 0})
    # balance_after = opening balance + running sum of signed amounts, per customer (card purchases do not move it)
    tx["balance_after"] = (opening[tx["customer_id"].to_numpy() - 1] + tx.groupby("customer_id")["_s"].cumsum()).round(2)
    tx = tx.drop(columns=["_o", "_s"])

    ls = pd.concat(ls_parts, ignore_index=True).sort_values(["customer_id", "month"]).reset_index(drop=True)
    d_date = np.where(D < NEVER, month_ends[np.minimum(D, T - 1)].values, np.datetime64("NaT"))
    customers = pd.DataFrame({
        "customer_id": cid, "signup_date": pd.to_datetime(c["signup"]), "segment": c["seg"],
        "loan_amount": c["loan_amount"], "loan_start_date": pd.to_datetime(c["loan_start"]),
        "credit_limit": c["credit_limit"]})
    defaults = pd.DataFrame({"customer_id": cid, "default_date": pd.to_datetime(d_date)})
    return {"customers": customers, "transactions": tx, "loan_status": ls, "defaults": defaults}


def main() -> None:
    tables = generate()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(DATA_RAW / f"{name}.csv", index=False)
        print(f"wrote data/raw/{name}.csv  rows={len(df):,}  cols={list(df.columns)}")
    d = tables["defaults"]["default_date"].notna()
    print(f"customers that ever default: {d.sum()} of {len(d)} ({d.mean():.1%})")


if __name__ == "__main__":
    main()
