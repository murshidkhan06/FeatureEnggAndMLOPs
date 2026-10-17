"""Leakage-safe feature pipeline: raw tables -> one row per (customer, as_of_date).

THE ONE RULE
  A feature computed for as_of_date may only use data that was AVAILABLE on that date:
        available_at <= as_of_date
  Transactions are available the day they happen.  loan_status for month M is published on day 5 of month M+1.
  `build_snapshot_features()` therefore FILTERS THE RAW TABLES FIRST and only then computes anything.
  `assert_no_future_data()` is a seat-belt that stops the program if a future row ever slips through.

  The "future-perturbation test" (tests/test_leakage.py) proves the rule: we corrupt every row that is NOT yet
  available, rebuild the features and check that they are bit-for-bit identical.

Features (6-month window = the 6 calendar months ending at as_of; example as_of = 2023-03-31 -> Oct 2022 .. Mar 2023)

  average_balance           mean of balance_after over transactions in the last 90 days
  balance_trend             OLS slope of the 6 month-end balances (currency / month)
                            tiny example: balances [100, 80, 60] at x = 0, 1, 2
                              x_mean = 1, y_mean = 80
                              slope = sum((x-1)*(y-80)) / sum((x-1)^2) = ((-1)(20) + 0 + (1)(-20)) / (1 + 0 + 1) = -20
  salary_credit_frequency   number of salary credits in the window / 6      (6 credits -> 1.0, 3 credits -> 0.5)
  loan_payment_ratio        loan payments made / loan payments due          (5 paid, 6 due -> 0.8333)
  missed_payment_count      sum of missed_payment over the last 6 PUBLISHED loan_status months (M-6 .. M-1)
  credit_utilization        mean monthly card spend / credit_limit          (spend 6,000 /month, limit 30,000 -> 0.2)
  recent_transaction_count  number of transactions in the last 30 days
  income_stability          std / mean (coefficient of variation) of salary amounts in the window
                            tiny example: salaries [100, 110, 90] -> mean 100, std (n-1) = 10 -> 0.10
                            fewer than 2 salary credits -> undefined (NaN)   <- a realistic MNAR missing value
  account_age               whole months between signup_date and as_of

Derived features (computed by add_derived_features, the SAME function is used by the API -> no skew)
  util_x_missed = credit_utilization * missed_payment_count        (0.5 * 2 -> 1.0)
  balance_trend_norm = clip(balance_trend / (|average_balance| + 1), -1, 1)
  log_average_balance = log1p(max(average_balance, 0))             (log1p(99) = 4.605)

Run:  python -m src.features.build_features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import DATA_PROCESSED, DATA_PRODUCTION, DATA_RAW, REFERENCE_PATH, SNAPSHOTS_PATH
from src.features.definitions import (AVAILABILITY_LAG_DAYS, BALANCE_DAYS, BASE_FEATURES, DERIVED_FEATURES,
                                      HORIZON_MONTHS, MODEL_FEATURES, RECENT_DAYS, TARGET, WINDOW_MONTHS)


# ------------------------------------------------------------------------------------------------------------
# loading + the availability rule
# ------------------------------------------------------------------------------------------------------------
def load_raw(raw_dir=DATA_RAW) -> dict[str, pd.DataFrame]:
    """Read the four raw CSV tables with proper dtypes (dates are real dates, tables are sorted)."""
    customers = pd.read_csv(raw_dir / "customers.csv", parse_dates=["signup_date", "loan_start_date"])
    tx = pd.read_csv(raw_dir / "transactions.csv", parse_dates=["txn_date"])
    tx = tx.sort_values(["customer_id", "txn_date"], kind="stable").reset_index(drop=True)
    ls = pd.read_csv(raw_dir / "loan_status.csv", parse_dates=["month", "available_at"])
    defaults = pd.read_csv(raw_dir / "defaults.csv", parse_dates=["default_date"])
    return {"customers": customers, "transactions": tx, "loan_status": ls, "defaults": defaults}


def month_end_shift(as_of: pd.Timestamp, months: int) -> pd.Timestamp:
    """Month-end `months` months away.  month_end_shift(2023-03-31, -6) -> 2022-09-30 ; (2023-03-31, +6) -> 2023-09-30."""
    as_of = pd.Timestamp(as_of)
    return as_of if months == 0 else as_of + pd.offsets.MonthEnd(months)


def filter_available(df: pd.DataFrame, as_of: pd.Timestamp, date_col: str, lag_days: int = 0) -> pd.DataFrame:
    """THE AVAILABILITY RULE: keep rows with  date_col + lag_days <= as_of.

    Example: loan_status row for 2023-02 has available_at = 2023-03-05.
             as_of = 2023-02-28 -> dropped ; as_of = 2023-03-31 -> kept."""
    return df[df[date_col] + pd.Timedelta(days=lag_days) <= pd.Timestamp(as_of)]


def assert_no_future_data(df: pd.DataFrame, as_of: pd.Timestamp, date_col: str) -> None:
    """Seat-belt: raise if ANY row is dated after as_of.  Called on every table inside the feature builder."""
    if len(df) and df[date_col].max() > pd.Timestamp(as_of):
        raise AssertionError(f"future data detected: max({date_col}) = {df[date_col].max().date()} > as_of {pd.Timestamp(as_of).date()}")


# ------------------------------------------------------------------------------------------------------------
# the snapshot builder
# ------------------------------------------------------------------------------------------------------------
def _slope_6(y: np.ndarray) -> np.ndarray:
    """Row-wise OLS slope against x = 0..5.   slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)."""
    x = np.arange(y.shape[1], dtype=float)
    xc = x - x.mean()
    return ((y - y.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc ** 2).sum()


def build_snapshot_features(customers: pd.DataFrame, transactions: pd.DataFrame, loan_status: pd.DataFrame,
                            as_of_date, defaults: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the 9 base features for every ELIGIBLE customer on as_of_date, using ONLY data available then.

    Eligible = signed up, at least 6 full months of transaction history, and not already in default at as_of.
    Returns columns: customer_id, as_of_date, <BASE_FEATURES>.
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    if as_of != as_of + pd.offsets.MonthEnd(0):
        raise ValueError("as_of_date must be a month-end, e.g. 2023-03-31")

    # 1) FILTER FIRST - everything below only ever sees available data
    tx = filter_available(transactions, as_of, "txn_date", AVAILABILITY_LAG_DAYS["transactions"])
    ls = filter_available(loan_status, as_of, "available_at", 0)       # available_at already includes the 5-day lag
    cu = customers[customers["signup_date"] <= as_of]
    assert_no_future_data(tx, as_of, "txn_date")
    assert_no_future_data(ls, as_of, "available_at")
    assert_no_future_data(cu, as_of, "signup_date")

    w_start = month_end_shift(as_of, -WINDOW_MONTHS)                     # window = (w_start, as_of]
    # 2) eligibility
    first_txn = tx.groupby("customer_id")["txn_date"].min()
    ok = first_txn[first_txn <= month_end_shift(as_of, -(WINDOW_MONTHS - 1))].index   # history starts in/before the 1st window month
    ids = cu.loc[cu["customer_id"].isin(ok), "customer_id"]
    if defaults is not None:
        already = defaults.loc[defaults["default_date"] <= as_of, "customer_id"]
        ids = ids[~ids.isin(already)]
    ids = pd.Index(sorted(ids), name="customer_id")
    cu = cu.set_index("customer_id").loc[ids]
    out = pd.DataFrame(index=ids)

    win = tx[(tx["txn_date"] > w_start) & tx["customer_id"].isin(ids)]
    # 3) salary based features
    sal = win[win["txn_type"] == "salary_credit"].groupby("customer_id")["amount"]
    out["salary_credit_frequency"] = (sal.size().reindex(ids).fillna(0) / WINDOW_MONTHS).round(4)
    cnt, mean, std = sal.size().reindex(ids), sal.mean().reindex(ids), sal.std().reindex(ids)   # std uses n-1
    out["income_stability"] = (std / mean).where(cnt >= 2).round(4)
    # 4) loan features
    paid = win[win["txn_type"] == "loan_payment"].groupby("customer_id").size().reindex(ids).fillna(0)
    win_month_ends = [month_end_shift(as_of, -k) for k in range(WINDOW_MONTHS)]
    loan_start_me = cu["loan_start_date"] + pd.offsets.MonthEnd(0)       # month-end of the loan's start month
    due = sum((loan_start_me < me).astype(int) for me in win_month_ends)
    out["loan_payment_ratio"] = np.where(due > 0, np.minimum(paid / due.clip(lower=1), 1.0), 1.0).round(4)
    ls_w = ls[(ls["month"] > month_end_shift(as_of, -(WINDOW_MONTHS + 1))) & ls["customer_id"].isin(ids)]
    out["missed_payment_count"] = ls_w.groupby("customer_id")["missed_payment"].sum().reindex(ids).fillna(0).astype(int)
    # 5) card utilisation (NaN for customers without a card)
    card = win[win["txn_type"] == "credit_card_spend"].groupby("customer_id")["amount"].sum().reindex(ids).fillna(0)
    out["credit_utilization"] = (card / WINDOW_MONTHS / cu["credit_limit"]).round(4)
    # 6) activity + balance features
    recent = tx[(tx["txn_date"] > as_of - pd.Timedelta(days=RECENT_DAYS)) & tx["customer_id"].isin(ids)]
    out["recent_transaction_count"] = recent.groupby("customer_id").size().reindex(ids).fillna(0).astype(int)
    bal90 = tx[(tx["txn_date"] > as_of - pd.Timedelta(days=BALANCE_DAYS)) & tx["customer_id"].isin(ids)]
    out["average_balance"] = bal90.groupby("customer_id")["balance_after"].mean().reindex(ids).round(2)
    out["balance_trend"] = _balance_trend(tx, ids, as_of)
    # 7) account age (whole months)
    out["account_age"] = (as_of.year - cu["signup_date"].dt.year) * 12 + (as_of.month - cu["signup_date"].dt.month)

    out = out.reset_index()
    out.insert(1, "as_of_date", as_of)
    return out[["customer_id", "as_of_date"] + BASE_FEATURES]


def _balance_trend(tx: pd.DataFrame, ids: pd.Index, as_of: pd.Timestamp) -> pd.Series:
    """Slope of the six month-end balances.  Months without transactions carry the previous balance forward."""
    lo = month_end_shift(as_of, -(WINDOW_MONTHS + 12))
    t = tx[(tx["txn_date"] > lo) & tx["customer_id"].isin(ids)]
    midx = t["txn_date"].dt.year * 12 + t["txn_date"].dt.month
    last = t.groupby(["customer_id", midx])["balance_after"].last().unstack()      # customer x month, last balance seen
    cols = list(range(as_of.year * 12 + as_of.month - WINDOW_MONTHS + 1 - 12, as_of.year * 12 + as_of.month + 1))
    last = last.reindex(index=ids, columns=cols).ffill(axis=1)                      # carry forward (past only)
    y = last.iloc[:, -WINDOW_MONTHS:].bfill(axis=1).to_numpy(dtype=float)           # bfill only inside the window
    return pd.Series(_slope_6(y).round(2), index=ids)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 3 derived features.  Used by training AND by the API (request_to_frame) -> identical logic online/offline."""
    out = df.copy()
    out["util_x_missed"] = out["credit_utilization"] * out["missed_payment_count"]
    out["balance_trend_norm"] = (out["balance_trend"] / (out["average_balance"].abs() + 1.0)).clip(-1.0, 1.0)
    out["log_average_balance"] = np.log1p(out["average_balance"].clip(lower=0))
    return out


# ------------------------------------------------------------------------------------------------------------
# target + split
# ------------------------------------------------------------------------------------------------------------
def add_target(snap: pd.DataFrame, defaults: pd.DataFrame) -> pd.DataFrame:
    """default_next_6m = 1 if default_date is in (as_of, as_of + 6 months].   (The label is built from the FUTURE on purpose.)

    Example: as_of 2023-03-31, default_date 2023-08-31 -> 1 ; default_date 2023-10-31 -> 0 ; no default -> 0."""
    d = snap.merge(defaults, on="customer_id", how="left")
    end = d["as_of_date"] + pd.offsets.MonthEnd(HORIZON_MONTHS)
    d[TARGET] = ((d["default_date"] > d["as_of_date"]) & (d["default_date"] <= end)).astype(int)
    return d.drop(columns="default_date")


def assign_split(as_of: pd.Series, splits: dict) -> pd.Series:
    """Time-based split on as_of_date (never random: customers repeat across snapshots and time order matters).

    train | purge (labels of train rows not yet known) | val | test | production (later period, includes the macro stress)."""
    a = pd.to_datetime(as_of)
    s = pd.Series("purge", index=a.index)
    s[a <= pd.Timestamp(splits["train_end"])] = "train"
    s[(a >= pd.Timestamp(splits["val_start"])) & (a <= pd.Timestamp(splits["val_end"]))] = "val"
    s[(a >= pd.Timestamp(splits["test_start"])) & (a <= pd.Timestamp(splits["test_end"]))] = "test"
    s[a >= pd.Timestamp(splits["production_start"])] = "production"
    return s


def build_all_snapshots(raw: dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
    """Build every monthly snapshot from first_as_of to last_as_of (labelled)."""
    sp = params["splits"]
    dates = pd.date_range(sp["first_as_of"], sp["last_as_of"], freq="ME")
    frames = [build_snapshot_features(raw["customers"], raw["transactions"], raw["loan_status"], d, raw["defaults"])
              for d in dates]
    snap = add_target(pd.concat(frames, ignore_index=True), raw["defaults"])
    snap["split"] = assign_split(snap["as_of_date"], sp)
    return snap


def write_production_batches(snap: pd.DataFrame, params: dict, seed: int = 11) -> None:
    """Write the three 'production' files the monitoring demos use.  Columns = API request fields + the TRUE target
    (available only months later in real life; we keep it so performance can be measured afterwards).

      batch_normal.csv    a calm month (no stress yet)
      batch_drifted.csv   a month inside the macro stress
      batch_dq_issue.csv  the calm month again, but a source-system migration made salary_credit_frequency empty
                          for ~38 % of customers (a DATA QUALITY problem, not drift)
    """
    DATA_PRODUCTION.mkdir(parents=True, exist_ok=True)
    cols = ["customer_id", "as_of_date"] + BASE_FEATURES + [TARGET]
    for name, key in (("batch_normal", "normal_as_of"), ("batch_drifted", "drifted_as_of")):
        b = snap[snap["as_of_date"] == pd.Timestamp(params["production"][key])][cols].rename(columns={"as_of_date": "features_as_of"})
        b.to_csv(DATA_PRODUCTION / f"{name}.csv", index=False)
    dq = pd.read_csv(DATA_PRODUCTION / "batch_normal.csv")
    rng = np.random.default_rng(seed)
    dq.loc[rng.random(len(dq)) < 0.38, "salary_credit_frequency"] = np.nan
    dq.to_csv(DATA_PRODUCTION / "batch_dq_issue.csv", index=False)


def main() -> None:
    from src.common.config import load_params
    params = load_params()
    raw = load_raw()
    snap = build_all_snapshots(raw, params)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    snap.to_csv(SNAPSHOTS_PATH, index=False)
    train = add_derived_features(snap[snap["split"] == "train"])
    train[MODEL_FEATURES].sample(params["reference_sample_rows"], random_state=0).to_csv(REFERENCE_PATH, index=False)
    write_production_batches(snap, params)
    print(f"snapshots: {len(snap):,} rows x {snap.shape[1]} cols -> {SNAPSHOTS_PATH.relative_to(SNAPSHOTS_PATH.parents[2])}")
    summary = snap.groupby("split").agg(rows=(TARGET, "size"), positives=(TARGET, "sum"), rate=(TARGET, "mean"),
                                        first=("as_of_date", "min"), last=("as_of_date", "max"))
    summary["rate"] = summary["rate"].round(4)
    print(summary.to_string())
    print(f"reference sample: {params['reference_sample_rows']} training rows -> data/processed/reference.csv")
    print("production batches: data/production/batch_normal.csv, batch_drifted.csv, batch_dq_issue.csv")


if __name__ == "__main__":
    main()
