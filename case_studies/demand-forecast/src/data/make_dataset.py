"""Deterministic synthetic retail demand generator (seeded - everybody gets identical data).

WHY synthetic?  Real retail data is private.  A generator lets us INJECT the real-world problems we
want to teach (and know the truth about them).  Nothing is downloaded.

What is generated
-----------------
8 products x 3 stores = 24 series, one row per series per day.
  * TRAINING period   2023-01-01 .. 2024-12-31   (731 days -> 17,544 rows)     -> data/raw/sales_daily.csv
  * PRODUCTION period 2025-01-01 .. 2025-02-28   (59 days  -> 1,416 rows)      -> data/production/batch_*.csv
        batch_normal.csv   business as usual (the training trend simply continues)
        batch_drifted.csv  the TEACHING TREND SHIFT: level jumps from ~130 to 150, then 170 -> 190 -> 210 in two months (four steps),
                           list prices +12% from 15 Jan, twice as many promotions
        batch_festival.csv Diwali lead-up 2024-10-20..2024-11-10 (an EXPECTED seasonal shift, not a failure)

How one day of demand is built (multiplicative - real demand scales with the size of the series)
------------------------------------------------------------------------------------------------
  mu = level(t) x product_scale x store_scale x weekly(dow) x annual(day of year) x holiday x festival_ramp
       x promotion_uplift x list_price_effect x temperature_effect
  sales = Poisson( mu x lognormal noise )          # counts are integers; noise grows with size

  Tiny example: level 100, product_scale 1.3, store_scale 1.2, Saturday 1.15, promo with 20% discount on a
  product with elasticity 1.8 (uplift 1 + 0.10 + 1.8*0.2 = 1.46):  mu = 100*1.3*1.2*1.15*1.46 = 262 units.

Real-world problems injected (documented in docs/DATASET.md)
------------------------------------------------------------
  * trend (level 100 -> 130) + weekly + annual seasonality + holiday spikes + Diwali lead-up ramp
  * promotions (price cuts) with product-specific elasticity; list-price steps (inflation)
  * stock-out zeros (sales = 0 although demand existed) and rare bulk-order outliers (x2.5..x4)
  * POS outage days: whole store-day has sales = NaN (missing completely at random)
  * weather-service outage: forecast_temp = NaN for a whole store-day (missing completely at random)
  * ACTUAL temperature is recorded after the fact; only the FORECAST is available at prediction time
    (using actual temperature as a feature is leakage - see leakage_demo.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import DATA_PRODUCTION, DATA_RAW, HISTORY_DRIFTED_PATH, HISTORY_PATH, REFERENCE_PATH
from src.features.calendar import DIWALI_DATES, HOLIDAYS
from src.features.definitions import BATCH_COLUMNS, HISTORY_COLUMNS, RAW_COLUMNS

SEED = 42
TRAIN_START, TRAIN_END = "2023-01-01", "2024-12-31"
PROD_START, PROD_END = "2025-01-01", "2025-02-28"
FESTIVAL_START, FESTIVAL_END = "2024-10-20", "2024-11-10"

# product_scale averages to exactly 1.0 over the 8 products, store_scale over the 3 stores.
PRODUCTS = {
    #        name                scale list_price temp_k  annual_amp peak_doy weekend_amp holiday_sens promo_elast
    "P01": dict(name="Cola 1L",           scale=1.30, list_price=45.0,  temp_k=0.25,  amp=0.12, peak=140, wk=1.3, hol=1.0, elast=2.0),
    "P02": dict(name="Ice Cream Tub",     scale=0.90, list_price=220.0, temp_k=0.45,  amp=0.30, peak=150, wk=1.5, hol=0.8, elast=2.0),
    "P03": dict(name="Instant Noodles",   scale=1.20, list_price=14.0,  temp_k=-0.05, amp=0.15, peak=20,  wk=1.0, hol=0.6, elast=2.5),
    "P04": dict(name="Milk 1L",           scale=1.60, list_price=56.0,  temp_k=0.0,   amp=0.04, peak=0,   wk=0.5, hol=0.4, elast=0.6),
    "P05": dict(name="Bread Loaf",        scale=1.40, list_price=40.0,  temp_k=0.0,   amp=0.03, peak=0,   wk=0.6, hol=0.3, elast=0.6),
    "P06": dict(name="Shampoo 340ml",     scale=0.55, list_price=285.0, temp_k=0.0,   amp=0.05, peak=300, wk=1.0, hol=0.8, elast=2.2),
    "P07": dict(name="Basmati Rice 5kg",  scale=0.45, list_price=480.0, temp_k=0.0,   amp=0.06, peak=310, wk=1.0, hol=1.2, elast=1.2),
    "P08": dict(name="Chocolate Box",     scale=0.60, list_price=350.0, temp_k=-0.08, amp=0.20, peak=340, wk=1.0, hol=1.8, elast=2.4),
}
STORES = {
    "S1": dict(city="Delhi",     scale=1.2, t_mean=25.0, t_amp=10.0),
    "S2": dict(city="Mumbai",    scale=1.0, t_mean=28.0, t_amp=3.5),
    "S3": dict(city="Bengaluru", scale=0.8, t_mean=24.0, t_amp=3.0),
}
# weekly shape (Mon..Sun), normalised to mean 1
_W = np.array([0.90, 0.88, 0.92, 0.97, 1.10, 1.25, 1.18])
WEEKLY = _W / _W.mean()

# average-level knots (design values the course talks about): training 100 -> 130, production 150 -> 210
TRAIN_KNOTS = {"2023-01-01": 100, "2023-09-01": 110, "2024-05-02": 120, "2024-12-31": 130}
NORMAL_KNOTS = {"2024-12-31": 130, "2025-02-28": 132}
# the shift arrives in four abrupt steps (e.g. a new distribution contract): 150 -> 170 -> 190 -> 210
DRIFT_STEPS = {"2025-01-01": 150, "2025-01-15": 170, "2025-01-29": 190, "2025-02-12": 210}


def level_path(dates: pd.DatetimeIndex, knots: dict[str, float], steps: bool = False) -> np.ndarray:
    """The 'average level' of demand.
    steps=False: piece-wise linear through the knots (np.interp over day numbers).
        Example: knots {'2023-01-01':100, '2023-09-01':110}: halfway between the two dates the level is 105.
    steps=True : a staircase - the level stays constant until the next date, then jumps.
        Example: {'2025-01-01':150, '2025-01-15':170}: 14 January -> 150, 15 January -> 170."""
    x = np.array([pd.Timestamp(k).value for k in knots], dtype=float)
    y = np.array(list(knots.values()), dtype=float)
    t = dates.values.astype("datetime64[ns]").astype("int64").astype(float)
    if steps:
        return y[np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(y) - 1)]
    return np.interp(t, x, y)


def list_price_factor(dates: pd.DatetimeIndex, shock: bool) -> np.ndarray:
    """Inflation: list prices rise 4% on 2023-07-01, 2024-01-01, 2024-07-01.  In the drifted scenario a
    further +12% shock arrives on 2025-01-15."""
    f = np.ones(len(dates))
    for d in ("2023-07-01", "2024-01-01", "2024-07-01"):
        f = f * np.where(dates >= pd.Timestamp(d), 1.04, 1.0)
    if shock:
        f = f * np.where(dates >= pd.Timestamp("2025-01-15"), 1.12, 1.0)
    return f


def make_weather(rng: np.random.Generator) -> pd.DataFrame:
    """Actual temperature + forecast temperature per store-day for the whole timeline (shared by all scenarios)."""
    dates = pd.date_range(TRAIN_START, PROD_END, freq="D")
    doy = dates.dayofyear.values
    rows = []
    for sid, s in STORES.items():
        ar = np.zeros(len(dates))
        for i in range(1, len(dates)):
            ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 1.8)          # day-to-day weather memory
        actual = s["t_mean"] + s["t_amp"] * np.sin(2 * np.pi * (doy - 110) / 365.25) + ar
        forecast = actual + rng.normal(0, 1.5, len(dates))        # a forecast is never perfect
        forecast[rng.random(len(dates)) < 0.015] = np.nan         # weather-API outage (MCAR)
        pos_out = rng.random(len(dates)) < 0.006                  # POS outage (MCAR)
        rows.append(pd.DataFrame(dict(date=dates, store_id=sid, temperature=actual.round(1),
                                      forecast_temp=forecast.round(1), pos_outage=pos_out)))
    return pd.concat(rows, ignore_index=True)


def promo_schedule(n: int, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random promotion campaigns of 3-6 days.  Returns (promotion flag, discount fraction).
    ~12% of campaigns are 'display only' (flag = 1 but no price cut)."""
    promo, disc = np.zeros(n), np.zeros(n)
    i = 0
    while i < n:
        if rng.random() < rate / 4.5:
            length = int(rng.integers(3, 7))
            d = 0.0 if rng.random() < 0.12 else float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]))
            promo[i:i + length] = 1
            disc[i:i + length] = d
            i += length + 1
        else:
            i += 1
    return promo, disc


def simulate_segment(dates: pd.DatetimeIndex, knots: dict, weather: pd.DataFrame, seed: int,
                     promo_rate: float = 0.10, price_shock: bool = False, steps: bool = False) -> pd.DataFrame:
    """Generate all 24 series for `dates`.  See the module docstring for the demand formula."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    level = level_path(dates, knots, steps)
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values
    lpf = list_price_factor(dates, price_shock)
    hol_strength = np.array([HOLIDAYS.get(d.strftime("%Y-%m-%d"), ("", 1.0))[1] for d in dates])
    is_hol = np.array([d.strftime("%Y-%m-%d") in HOLIDAYS for d in dates]).astype(int)
    ramp = np.zeros(n)                                   # Diwali lead-up: 7 days of rising demand
    for dd in DIWALI_DATES:
        for k in range(1, 8):
            idx = np.where(dates == dd - pd.Timedelta(days=k))[0]
            if len(idx):
                ramp[idx[0]] = (8 - k) / 7
    frames = []
    for pid, p in PRODUCTS.items():
        for sid, s in STORES.items():
            w = weather[weather.store_id == sid].set_index("date").loc[dates]
            promo, disc_frac = promo_schedule(n, promo_rate, rng)
            weekly = 1 + p["wk"] * (WEEKLY[dow] - 1)
            annual = 1 + p["amp"] * np.sin(2 * np.pi * (doy - p["peak"] + 91) / 365.25)
            holiday = 1 + (hol_strength - 1) * p["hol"]
            festive = 1 + 0.30 * p["hol"] * ramp
            uplift = np.where(promo == 1, 1 + 0.10 + p["elast"] * disc_frac, 1.0)
            price_eff = lpf ** -0.6
            temp_eff = 1 + p["temp_k"] * (w.temperature.values - 27.0) / 10.0
            mu = level * p["scale"] * s["scale"] * weekly * annual * holiday * festive * uplift * price_eff * temp_eff
            noisy = mu * np.exp(rng.normal(0, 0.08, n) - 0.0032)
            sales = rng.poisson(np.clip(noisy, 0.1, None)).astype(float)
            # outliers: rare bulk orders
            out = rng.random(n) < 0.003
            sales[out] = np.round(sales[out] * rng.uniform(2.5, 4.0, out.sum()))
            # stock-outs: runs of 1-3 days with zero sales although customers wanted the product
            i = 0
            while i < n:
                if rng.random() < 0.0015:
                    sales[i:i + int(rng.integers(1, 4))] = 0
                    i += 4
                else:
                    i += 1
            sales[w.pos_outage.values] = np.nan                    # POS outage: sales not recorded
            list_p = np.round(p["list_price"] * lpf, 2)
            discount = np.round(list_p * disc_frac, 2)
            frames.append(pd.DataFrame(dict(
                date=dates, product_id=pid, store_id=sid, sales=sales, price=np.round(list_p - discount, 2),
                discount=discount, promotion=promo.astype(int), holiday=is_hol,
                temperature=w.temperature.values, forecast_temp=w.forecast_temp.values,
                day_of_week=dow)))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "product_id", "store_id"]).reset_index(drop=True)[RAW_COLUMNS]


def generate_all(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Everything in memory: {'train','normal','drifted','festival'} (raw column layout)."""
    weather = make_weather(np.random.default_rng(seed))
    train = simulate_segment(pd.date_range(TRAIN_START, TRAIN_END), TRAIN_KNOTS, weather, seed + 1)
    prod_dates = pd.date_range(PROD_START, PROD_END)
    normal = simulate_segment(prod_dates, NORMAL_KNOTS, weather, seed + 2, promo_rate=0.10, price_shock=False)
    drifted = simulate_segment(prod_dates, DRIFT_STEPS, weather, seed + 3, promo_rate=0.18, price_shock=True, steps=True)
    festival = train[(train.date >= FESTIVAL_START) & (train.date <= FESTIVAL_END)].copy()
    return dict(train=train, normal=normal, drifted=drifted, festival=festival)


def to_batch(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw layout -> production-batch layout (column `date` becomes the API field `forecast_date`)."""
    return raw.rename(columns={"date": "forecast_date"})[BATCH_COLUMNS].copy()


def from_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """Production-batch layout -> history layout (inverse of to_batch for the history columns)."""
    return batch.rename(columns={"forecast_date": "date"}).assign(date=lambda d: pd.to_datetime(d["date"]))


def write_all() -> dict[str, pd.DataFrame]:
    """Write every dataset file.  Run:  python -m src.data.make_dataset"""
    from src.features.build_features import compute_features   # local import: features need data only at call time
    from src.features.definitions import ALL_CANDIDATE_FEATURES, SPLIT, WARMUP_DAYS

    d = generate_all()
    for p in (DATA_RAW, DATA_PRODUCTION, REFERENCE_PATH.parent):
        p.mkdir(parents=True, exist_ok=True)
    d["train"].to_csv(DATA_RAW / "sales_daily.csv", index=False)
    for name in ("normal", "drifted", "festival"):
        to_batch(d[name]).to_csv(DATA_PRODUCTION / f"batch_{name}.csv", index=False)
    # history store snapshots (what the nightly pipeline would have loaded): training + production actuals
    d["train"].pipe(lambda t: pd.concat([t, d["normal"]]))[HISTORY_COLUMNS].to_csv(HISTORY_PATH, index=False)
    d["train"].pipe(lambda t: pd.concat([t, d["drifted"]]))[HISTORY_COLUMNS].to_csv(HISTORY_DRIFTED_PATH, index=False)
    # reference sample (training-time model INPUT frame, TRAIN split only)
    feats = compute_features(d["train"])
    feats = feats[(feats.history_days >= WARMUP_DAYS) & feats.sales.notna()]
    ref = feats[feats.date <= pd.Timestamp(SPLIT["train_end"])]
    ref[["date"] + ALL_CANDIDATE_FEATURES + ["sales"]].sample(5000, random_state=0).sort_values("date").to_csv(
        REFERENCE_PATH, index=False)
    print(f"train rows={len(d['train'])}  production rows={len(d['normal'])}  reference rows=5000")
    return d


if __name__ == "__main__":
    write_all()
