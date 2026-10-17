# Data dictionary - Demand Forecasting (synthetic retail data)

Everything is **generated** by `python -m src.data.make_dataset` (seed 42, no downloads, ~1 second). The generator is a *story we control*: because we
know how demand was created, we can tell students exactly which problem was injected and check whether a method finds it.

## 1. What is generated

| File | Rows | Meaning |
|---|---|---|
| `data/raw/sales_daily.csv` | 17,544 | TRAINING period 2023-01-01 .. 2024-12-31 (731 days) x 24 series (8 products x 3 stores) |
| `data/production/batch_normal.csv` | 1,416 | production period 2025-01-01 .. 2025-02-28 (59 days), business as usual (trend simply continues) |
| `data/production/batch_drifted.csv` | 1,416 | same period with the **teaching trend shift**: level 150 -> 170 -> 190 -> 210 (four steps), list prices +12% from 15 Jan, twice as many promotions |
| `data/production/batch_festival.csv` | 528 | Diwali lead-up 2024-10-20 .. 2024-11-10 (copied from the training data): an **expected seasonal shift** |
| `data/processed/sales_history.csv` | 18,960 | the "online feature store" stand-in read by the API: training + normal production actuals |
| `data/processed/sales_history_drifted.csv` | 18,960 | the same, but with the drifted production actuals |
| `data/processed/reference.csv` | 5,000 | sample of the TRAINING-time model input frame (train split only) + `sales`: the reference for drift tests |
| `data/processed/features.csv` | 16,744 | engineered training frame (`python -m src.features.build_features`) |

A **series** is one `(product_id, store_id)` pair. Weekly average sales per series-day (design levels): training 100 -> 130, production normal ~130,
production drifted 150 -> 210 (`docs/sample_outputs/01_data_profile.txt` shows the measured weekly table).

## 2. Raw columns

| Column | Type | Meaning | Available when we forecast day D+1 at the end of day D? |
|---|---|---|---|
| `date` | date | calendar day | yes (it is the forecast day) |
| `product_id` | str P01..P08 | product (Cola, Ice Cream, Noodles, Milk, Bread, Shampoo, Rice, Chocolate) | yes |
| `store_id` | str S1..S3 | store (Delhi, Mumbai, Bengaluru) | yes |
| `sales` | count | **TARGET**: units sold that day | **no** for the forecast day; yes for earlier days |
| `price` | INR | shelf price after discount | yes - planned price |
| `discount` | INR | discount per unit already inside `price` | yes - planned |
| `promotion` | 0/1 | a promotion campaign runs | yes - planned |
| `holiday` | 0/1 | retail holiday (calendar published a year ahead) | yes |
| `temperature` | deg C | **actual** temperature, recorded after the day | **no -> leakage** (Rejected feature) |
| `forecast_temp` | deg C | weather **forecast** for that day | yes (may be missing) |
| `day_of_week` | 0-6 | Monday = 0 | yes |

Production batches use `forecast_date` instead of `date` and contain the TRUE `sales` (labels arrive the next day in retail, so performance can be
measured after the fact) plus the actual `temperature` (kept only to show it is never used).

## 3. Engineered features (the model-input frame)
Full metadata (definition, source, owner, calculation logic, availability, status, version) lives in `configs/feature_registry.yaml`.

| Feature | Formula | Why | Status |
|---|---|---|---|
| `lag_1`, `lag_7`, `lag_14` | `sales.shift(k)` per series | yesterday / same weekday last week / two weeks ago | Approved |
| `rolling_mean_7`, `rolling_mean_30` | `sales.shift(1).rolling(w).mean()` | recent level; **shift first** so day t never sees day t | Approved |
| `day_of_week`, `week_of_year`, `month`, `is_weekend`, `is_holiday` | calendar | weekly / annual seasonality | Approved |
| `dow_sin/dow_cos`, `month_sin/month_cos` | `sin/cos(2*pi*x/period)` | cyclical encoding: Sunday is next to Monday | Approved |
| `trend_index` | days since 2023-01-01 | lets a linear model learn a slope (dropped by selection) | Approved |
| `price`, `price_change`, `discount_pct`, `promotion_flag` | `price/price_yesterday-1`, `discount/(price+discount)` | price and promotion effects | Approved |
| `forecast_temp` | weather forecast | weak driver for ice cream / cola | Approved |
| `product_id`, `store_id` | one-hot inside the pipeline | series identity | Approved |
| `temperature` | actual weather | **not known at prediction time** | **Rejected** (leakage) |
| `rolling_mean_7_centered`, `rolling_mean_7_noshift`, `sales_today` | leaky demo columns | exist only in `leakage_demo.py` | **Rejected** |

`configs/selected_features.yaml` is the output of the selection step (14 features, chosen on **training rows only**).

## 4. Problems injected on purpose (and where the course meets them)

| Problem | How it is generated | Why it matters / where handled |
|---|---|---|
| **Trend** | level 100 -> 130 over training; production 150 -> 210 | tree models cannot extrapolate (`14_trend_shift_extrapolation.txt`); retraining (`12_retrain_promotion.txt`) |
| **Weekly seasonality** | weekend lift per product | `day_of_week`, seasonal-naive baseline |
| **Annual seasonality** | sine with product-specific peak (ice cream in May, shampoo in winter ...) | month sin/cos; seasonal drift reference |
| **Holidays + Diwali ramp** | demand multipliers; 7-day lead-up ramp before Diwali | expected drift (`11b`, `11c`) |
| **Promotions** | 3-6 day campaigns; uplift = 1 + 0.10 + elasticity x discount; ~12% are "display only" (flag 1, no price cut) | `promotion_flag`, `discount_pct` |
| **Price effect + inflation** | list price +4% three times in training; +12% shock in the drifted batch | price drift, unit-fragile absolute price (dropped), `price_change` |
| **Noise** | lognormal multiplicative noise + Poisson counts | irreducible error: WAPE ~0.12 is a floor, not a bug |
| **Stock-out zeros** | runs of 1-3 days with `sales = 0` although demand existed (42 rows) | MAPE = inf; WAPE is defined (`13_failure_scenarios.txt`) |
| **Outliers** | 0.3% bulk orders x2.5..x4 | robust evaluation; they stay in the data (real life) |
| **Missing `sales` (MCAR)** | POS outage: whole store-day NaN (0.46%) | the calendar is completed (NaN row), lag falls back to the recent average; row dropped from training |
| **Missing `forecast_temp` (MCAR)** | weather-service outage (1.5%) | imputed (median) + missing indicator inside the pipeline |
| **Correlated features** | lags and rolling means are strongly correlated (e.g. 0.97 between rolling_mean_7 and rolling_mean_30) | redundancy filter in selection; permutation importance caveat |
| **Leakage columns** | `temperature` (actual), centred / un-shifted rolling windows | feature registry gate, `04_leakage_demo.txt` |
| **Duplicated dates, unit change** | created on demand by `failure_scenarios.py` | data-quality checks (`src/features/data_quality.py`) |

## 5. What is NOT in the data (limits)
No real customers, no stock levels (so a stock-out looks like zero demand), no competitor prices, no future promotion calendar beyond what a request states.
Numbers are *plausible*, not real; do not quote them as facts about retail.
