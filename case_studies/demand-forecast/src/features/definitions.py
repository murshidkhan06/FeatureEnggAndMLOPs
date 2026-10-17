"""Single source of truth for column names, feature groups and split dates.

Keeping the lists here (and not scattered through the code) means the training code, the batch
forecaster, the API and the tests can never disagree about what a "feature" is.

Vocabulary
----------
series          one (product_id, store_id) pair.  8 products x 3 stores = 24 series.
target          sales = units sold that day.
horizon         how many days ahead we forecast.  Horizon 1 = "made at the end of day D for day D+1".
known in advance   information we really have at forecast time (calendar, planned price/promo, weather FORECAST).
"""
from __future__ import annotations

import pandas as pd

KEYS = ["product_id", "store_id"]
TARGET = "sales"
TREND_ORIGIN = pd.Timestamp("2023-01-01")          # trend_index = days since this date
WARMUP_DAYS = 30                                   # first 30 days of a series cannot have a full rolling_mean_30

# ---- raw columns --------------------------------------------------------------------------
RAW_COLUMNS = ["date", "product_id", "store_id", "sales", "price", "discount", "promotion", "holiday",
               "temperature", "forecast_temp", "day_of_week"]
HISTORY_COLUMNS = ["date", "product_id", "store_id", "sales", "price", "discount", "promotion", "holiday",
                   "forecast_temp"]             # what the history store keeps (NO actual temperature)
BATCH_COLUMNS = ["forecast_date", "product_id", "store_id", "price", "discount", "promotion", "holiday",
                 "forecast_temp", "sales", "temperature"]   # API request fields + TRUE target + actual weather

# ---- engineered feature groups (the model-input frame) --------------------------------------
CALENDAR = ["day_of_week", "week_of_year", "month", "is_weekend", "is_holiday", "trend_index"]
CYCLICAL = ["dow_sin", "dow_cos", "month_sin", "month_cos"]
LAGS = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]        # "level" features: they carry the scale
PRICE_PROMO = ["price", "discount_pct", "price_change", "promotion_flag"]
WEATHER = ["forecast_temp"]
IDENTITY = ["product_id", "store_id"]                                            # categorical series identity

ALL_CANDIDATE_FEATURES = IDENTITY + CALENDAR + CYCLICAL + LAGS + PRICE_PROMO + WEATHER
LEVEL_FEATURES = LAGS + ["trend_index"]               # explain "how big is this series right now"
PATTERN_FEATURES = [f for f in ALL_CANDIDATE_FEATURES if f not in LEVEL_FEATURES + IDENTITY]  # explain "today vs normal"
# features that are always kept: business knowledge says they matter even if a statistic is lukewarm
DOMAIN_KEEP = ["product_id", "store_id", "is_holiday", "promotion_flag", "discount_pct", "day_of_week"]

# ---- features that are LEAKY and must never be used (they exist only for the lessons) ---------
LEAKY_FEATURES = ["temperature", "rolling_mean_7_centered", "rolling_mean_7_noshift", "sales_today"]

# ---- default chronological split (can be overridden in params.yaml) --------------------------
SPLIT = dict(train_end="2024-06-30", valid_end="2024-09-30", test_end="2024-12-31")
