"""Hand calculation vs code for the main features.  Run:  python -m src.features.manual_examples

Every block: (1) the numbers a student can verify with a calculator, (2) the same number from compute_features().
If the two ever disagree the test-suite fails (tests/test_features.py re-uses these functions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.facts import save_fact
from src.common.paths import DATA_RAW, HISTORY_PATH
from src.features.build_features import compute_features, features_for_batch
from src.features.history_store import HistoryStore

SALES = [10, 12, 11, 13, 20, 22, 21, 14]          # days 1..8 of one tiny series


def toy_frame(sales=SALES, start="2025-01-01", drop_day: int | None = None) -> pd.DataFrame:
    n = len(sales)
    d = pd.DataFrame(dict(date=pd.date_range(start, periods=n), product_id="PX", store_id="SX", sales=np.array(sales, float),
                          price=40.0, discount=0.0, promotion=0, holiday=0, forecast_temp=25.0))
    if drop_day is not None:
        d = d.drop(index=drop_day - 1)
    return d


def main() -> None:
    facts: dict = {}
    print("=== 1. rolling_mean_7: shift(1) BEFORE rolling ===")
    print(f"sales days 1..8 = {SALES}")
    manual_ok = np.mean(SALES[0:7])                       # days 1..7
    manual_leak = np.mean(SALES[1:8])                     # days 2..8 (includes day 8 = today)
    f = compute_features(toy_frame())
    code_day8 = float(f.loc[7, "rolling_mean_7"])
    leaky_code = float(pd.Series(SALES, dtype=float).rolling(7).mean().iloc[7])
    print(f"hand : day 8 rolling_mean_7 = mean(10,12,11,13,20,22,21) = {manual_ok:.2f}   (days 1..7, today excluded)")
    print(f"code : compute_features day 8 rolling_mean_7             = {code_day8:.2f}   {'MATCH' if abs(code_day8 - manual_ok) < 1e-9 else 'MISMATCH'}")
    print(f"WRONG: rolling(7) without shift on day 8                 = {leaky_code:.2f}   = mean(12,11,13,20,22,21,14): includes day 8's own sales (14)")
    facts["rolling_mean_7"] = dict(sales=SALES, hand=round(manual_ok, 2), code=round(code_day8, 2), wrong_no_shift=round(leaky_code, 2))

    print("\n=== 2. lags ===")
    print(f"hand : day 8 lag_1 = sales(day 7) = 21 ; lag_7 = sales(day 1) = 10")
    print(f"code : lag_1 = {f.loc[7, 'lag_1']:.0f} ; lag_7 = {f.loc[7, 'lag_7']:.0f}")
    facts["lags"] = dict(lag_1=float(f.loc[7, "lag_1"]), lag_7=float(f.loc[7, "lag_7"]))

    print("\n=== 3. a missing day (POS outage) - calendar completion + fallback ===")
    g = compute_features(toy_frame(drop_day=7))            # day 7 missing
    r = g[g.date == g.date.max()].iloc[0]
    print("sales for day 7 never arrived.  For day 8: lag_1 would be NaN -> filled with rolling_mean_7")
    print(f"hand : rolling_mean_7 = mean(10,12,11,13,20,22) = {np.mean([10, 12, 11, 13, 20, 22]):.2f} (6 valid days, min_periods=4 -> ok)")
    print(f"code : lag_1 = {r.lag_1:.2f}, rolling_mean_7 = {r.rolling_mean_7:.2f}   (shift means 'previous CALENDAR day', not 'previous row')")
    facts["missing_day"] = dict(lag_1=round(float(r.lag_1), 2), rolling_mean_7=round(float(r.rolling_mean_7), 2))

    print("\n=== 4. price_change and discount_pct ===")
    t = toy_frame()
    t.loc[7, ["price", "discount"]] = [38.0, 2.0]          # day 8: price 40 -> 38, discount 2 INR (list price 40)
    t.loc[7, "promotion"] = 1
    h = compute_features(t).iloc[7]
    print(f"hand : price_change = 38/40 - 1 = {38 / 40 - 1:+.3f} ; discount_pct = 2/(38+2) = {2 / 40:.3f}")
    print(f"code : price_change = {h.price_change:+.3f} ; discount_pct = {h.discount_pct:.3f} ; promotion_flag = {h.promotion_flag}")
    facts["price"] = dict(price_change=round(float(h.price_change), 3), discount_pct=round(float(h.discount_pct), 3))

    print("\n=== 5. cyclical encoding (optional): sin/cos of the weekday ===")
    for name, k in (("Monday", 0), ("Thursday", 3), ("Sunday", 6)):
        print(f"  {name:<9} day_of_week={k}: sin={np.sin(2 * np.pi * k / 7):+.3f}  cos={np.cos(2 * np.pi * k / 7):+.3f}")
    d_sun_mon = np.hypot(np.sin(2 * np.pi * 6 / 7) - np.sin(0), np.cos(2 * np.pi * 6 / 7) - np.cos(0))
    d_mon_thu = np.hypot(np.sin(2 * np.pi * 3 / 7) - np.sin(0), np.cos(2 * np.pi * 3 / 7) - np.cos(0))
    print(f"  distance Sunday->Monday = {d_sun_mon:.3f}  (neighbours) vs Monday->Thursday = {d_mon_thu:.3f}.  As plain numbers 6 and 0")
    print("  look 6 apart; on the circle they are adjacent.  December/January work the same with month_sin/month_cos.")
    facts["cyclical"] = dict(sun_to_mon=round(float(d_sun_mon), 3), mon_to_thu=round(float(d_mon_thu), 3))

    print("\n=== 6. a REAL row, recomputed by hand from the raw history ===")
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    key = ("P04", "S2")
    day = pd.Timestamp("2024-11-01")                          # Diwali 2024
    s = raw[(raw.product_id == key[0]) & (raw.store_id == key[1])].set_index("date")["sales"]
    prev7 = s[day - pd.Timedelta(days=7): day - pd.Timedelta(days=1)]
    hand = dict(lag_1=s[day - pd.Timedelta(days=1)], lag_7=s[day - pd.Timedelta(days=7)], rolling_mean_7=prev7.mean())
    feats = compute_features(raw[(raw.product_id == key[0]) & (raw.store_id == key[1])])
    row = feats[feats.date == day].iloc[0]
    print(f"series P04 (milk) / S2 (Mumbai), day {day.date()} (Diwali, is_holiday={row.is_holiday}), actual sales = {s[day]:.0f}")
    print(f"previous 7 days: {prev7.astype(int).tolist()}")
    for k in ("lag_1", "lag_7", "rolling_mean_7"):
        print(f"  {k:<15} hand = {hand[k]:8.2f}   code = {row[k]:8.2f}   {'MATCH' if abs(hand[k] - row[k]) < 1e-6 else 'MISMATCH'}")
    facts["real_row"] = dict(day=str(day.date()), hand={k: round(float(v), 2) for k, v in hand.items()},
                             code={k: round(float(row[k]), 2) for k in hand})

    print("\n=== 7. offline (training) vs online (history store) - same function, same numbers ===")
    store = HistoryStore.from_csv(HISTORY_PATH)
    on = store.features_for("P04", "S2", day, price=float(row.price), discount=float(row.discount), promotion=int(row.promotion),
                            holiday=int(row.holiday), forecast_temp=None if pd.isna(row.forecast_temp) else float(row.forecast_temp))
    cols = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30", "price_change", "discount_pct"]
    for c in cols:
        print(f"  {c:<16} offline = {row[c]:9.3f}   online = {on.iloc[0][c]:9.3f}")
    ok = np.allclose(row[cols].astype(float).values, on.iloc[0][cols].astype(float).values)
    print("  PARITY:", "OK (no training-serving skew)" if ok else "MISMATCH")
    facts["parity"] = dict(ok=bool(ok))
    save_fact("feature_manual_examples", facts)


if __name__ == "__main__":
    main()
