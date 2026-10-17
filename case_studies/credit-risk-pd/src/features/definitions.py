"""Feature DEFINITIONS shared by training, batch scoring and the API.

Keeping the names, windows and availability lags in ONE file means the offline (training) code path and the
online (serving) code path can never silently disagree -> no training-serving skew.

Vocabulary
  as_of_date (prediction timestamp) : the month-end on which we score customers, e.g. 2023-03-31.
  window                            : how far back a feature looks, e.g. 6 months.
  availability lag                  : days between the event and the moment the bank can actually USE the data.
  availability rule                 : a row may be used only if  available_at <= as_of_date.

Worked example (availability rule)
  loan_status for month 2023-02 is published on day 5 of the next month -> available_at = 2023-02-28 + 5 days = 2023-03-05.
  Scoring as_of 2023-02-28 : 2023-03-05 <= 2023-02-28 ?  NO  -> the February row must NOT be used.
  Scoring as_of 2023-03-31 : 2023-03-05 <= 2023-03-31 ?  YES -> the February row may be used.
"""
from __future__ import annotations

TARGET = "default_next_6m"
HORIZON_MONTHS = 6                 # we predict default within the next 6 months
WINDOW_MONTHS = 6                  # look-back for most behavioural features
BALANCE_DAYS = 90                  # look-back for average_balance
RECENT_DAYS = 30                   # look-back for recent_transaction_count

# availability lag in days per source table (0 = usable the same day)
AVAILABILITY_LAG_DAYS = {"transactions": 0, "loan_status": 5, "customers": 0}

# --- approved, leakage-safe features --------------------------------------------------------------
BASE_FEATURES = [
    "average_balance",           # mean balance over the last 90 days
    "balance_trend",             # slope of month-end balances over 6 months (currency units per month)
    "salary_credit_frequency",   # salary credits per month over 6 months
    "loan_payment_ratio",        # loan payments made / loan payments due over 6 months
    "missed_payment_count",      # missed payments in the last 6 PUBLISHED loan_status months
    "credit_utilization",        # mean over 6 months of (monthly card spend / credit limit)
    "recent_transaction_count",  # number of transactions in the last 30 days
    "income_stability",          # coefficient of variation (std / mean) of salary amounts over 6 months
    "account_age",               # months since signup
]
DERIVED_FEATURES = [             # computed from BASE_FEATURES by add_derived_features() (same code offline + online)
    "util_x_missed",             # credit_utilization * missed_payment_count  (interaction)
    "balance_trend_norm",        # balance_trend / (|average_balance| + 1)    (scale-free trend, clipped)
    "log_average_balance",       # log1p(max(average_balance, 0))             (tames the long right tail)
]
MODEL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

# --- features that exist ONLY to demonstrate leakage; they are Rejected in the registry --------------
LEAKY_FEATURES = [
    "average_balance_leaky",     # balance averaged up to the label window -> includes post-default collapse
    "dpd_target_period",         # max days_past_due INSIDE the 6-month target window
    "recoveries_target_period",  # recoveries collected INSIDE the 6-month target window
    "collections_status_latest", # collections status at the data-extract date (after the outcome)
]

ID_COLUMNS = ["customer_id", "as_of_date"]
