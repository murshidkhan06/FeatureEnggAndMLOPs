"""FOUR THINGS THAT GO WRONG IN REAL FORECASTING SYSTEMS - reproduced on purpose.   python -m src.monitoring.failure_scenarios

  (a) STALE HISTORY STORE   the nightly refresh did not run; lags come from old days -> forecasts are stuck on the past
  (b) MAPE EXPLODES         zero-sales (stock-out) days make a percentage error infinite -> use WAPE
  (c) DUPLICATED DATES      a pipeline retry loaded last week twice -> rolling means are silently wrong -> data test
  (d) PRICE UNIT CHANGE     a source system starts sending paise instead of rupees (x100) -> data quality + API range check

For every scenario: SYMPTOM (what you would see on a dashboard), CAUSE, DETECTION (which test/check catches it) and FIX.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.artifact import load_artifact
from src.common.facts import save_fact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH, DATA_RAW
from src.features import data_quality
from src.features.build_features import check_history, compute_features, training_frame
from src.features.definitions import HISTORY_COLUMNS
from src.features.history_store import HistoryStore
from src.models import metrics as M
from src.models.pipelines import build_pipeline
from src.models.train import score, time_split


def stale_history(raw: pd.DataFrame, art: dict) -> dict:
    """Freeze the history store on Diwali 2024-11-01 and keep forecasting the days after the festival."""
    freeze = pd.Timestamp("2024-11-01")
    days = pd.date_range(freeze + pd.Timedelta(days=1), freeze + pd.Timedelta(days=4))
    hist = raw[HISTORY_COLUMNS]
    stale_store = HistoryStore(hist[hist["date"] <= freeze])
    rows = []
    for t in days:
        fresh_store = HistoryStore(hist[hist["date"] < t])
        today = raw[raw["date"] == t]
        pf, ps, act = [], [], []
        for r in today.itertuples():
            args = (r.product_id, r.store_id, t, r.price, r.discount, int(r.promotion), int(r.holiday),
                    None if pd.isna(r.forecast_temp) else r.forecast_temp)
            pf.append(score(art["pipeline"], fresh_store.features_for(*args), art["input_columns"])[0])
            ps.append(score(art["pipeline"], stale_store.features_for(*args), art["input_columns"])[0])
            act.append(r.sales)
        act = np.array(act, float)
        ok = ~np.isnan(act)
        rows.append(dict(date=str(t.date()), stale_days=(t - freeze).days - 1, actual_units=int(act[ok].sum()),
                         fresh_forecast=int(np.sum(np.array(pf)[ok])), stale_forecast=int(np.sum(np.array(ps)[ok])),
                         fresh_wape=round(M.wape(act[ok], np.array(pf)[ok]), 3), stale_wape=round(M.wape(act[ok], np.array(ps)[ok]), 3)))
    t = pd.DataFrame(rows)
    refused = None
    try:
        stale_store.features_for("P04", "S2", freeze + pd.Timedelta(days=6), 56.0)
    except ValueError as exc:
        refused = str(exc)[:110]
    return dict(table=t, refused=refused)


def zero_days(raw: pd.DataFrame) -> dict:
    """MAPE vs WAPE with stock-out zeros."""
    y, f = np.array([100.0, 0.0, 50.0]), np.array([90.0, 10.0, 55.0])
    tiny = dict(mape=M.mape_naive(y, f), wape=M.wape(y, f))
    z = raw[raw["sales"] == 0]
    real = dict(zero_rows=len(z), share=len(z) / len(raw))
    return dict(tiny=tiny, real=real)


def duplicated_dates(raw: pd.DataFrame) -> dict:
    """A retry loads the last 3 days of one series twice."""
    one = raw[(raw.product_id == "P04") & (raw.store_id == "S2")][HISTORY_COLUMNS].sort_values("date")
    clean = one.tail(40)
    dup = pd.concat([clean, clean.tail(3)]).sort_values("date", kind="stable").reset_index(drop=True)
    wrong = dup["sales"].shift(1).rolling(7).mean().iloc[-1]          # what a naive pipeline computes without a check
    right = clean["sales"].shift(1).rolling(7).mean().iloc[-1]
    err = None
    try:
        check_history(dup)
    except ValueError as exc:
        err = str(exc)
    dq = data_quality.validate(dup.rename(columns={"date": "forecast_date"}))
    return dict(rolling_wrong=float(wrong), rolling_right=float(right), check_history=err,
                dq_failed=[p for p in dq["problems"] if "duplicate" in p])


def price_unit_change(raw: pd.DataFrame, art: dict, params: dict) -> dict:
    """Rupees -> paise (x100) for one week."""
    batch = raw[(raw.date >= "2024-12-01") & (raw.date <= "2024-12-07")].rename(columns={"date": "forecast_date"}).copy()
    bad = batch.assign(price=batch["price"] * 100, discount=batch["discount"] * 100)
    dq = data_quality.validate(bad)
    sample = {"product_id": "P03", "store_id": "S1"}
    df = training_frame(raw)
    tr, va, te = time_split(df, params["split"])
    row = te[(te.product_id == "P03") & (te.store_id == "S1")].iloc[[0]].copy()
    row_bad = row.assign(price_change=99.0)          # the first day after the unit change: price / price_yesterday - 1 = 99
    out = {}
    for name, pipe in (("hgb (champion)", art["pipeline"]), ("ridge_log", build_pipeline("ridge_log", art["input_columns"], params)
                                                         .fit(pd.concat([tr, va])[art["input_columns"]], pd.concat([tr, va])["sales"]))):
        out[name] = (float(score(pipe, row, art["input_columns"])[0]), float(score(pipe, row_bad, art["input_columns"])[0]))
    return dict(dq_failed=[p for p in dq["problems"]], models=out, actual=float(row["sales"].iloc[0]), sample=sample)


def main() -> None:
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    art, params = load_artifact(CHAMPION_PATH), load_params()
    facts = {}

    print("=" * 100 + "\n(a) STALE HISTORY STORE - forecasts after Diwali with a store frozen on 2024-11-01 (the festival peak)\n" + "=" * 100)
    s = stale_history(raw, art)
    print(s["table"].to_string(index=False))
    t = s["table"]
    last = t.iloc[-1]
    over = last.stale_forecast / last.actual_units - 1
    print(f"\nSYMPTOM : accuracy decays with every missing day: stale WAPE {' -> '.join(f'{x:.3f}' for x in t.stale_wape)} "
          f"(fresh: {' -> '.join(f'{x:.3f}' for x in t.fresh_wape)}).  On {last.date} the stale forecast says {last.stale_forecast} units, "
          f"actual demand was {last.actual_units} ({over:+.0%}): it is still 'celebrating' the festival.")
    print("CAUSE   : lag_1/lag_7/rolling means still describe the festival days; the model has no idea the festival is over.")
    print("DETECT  : `history_stale_days` in every API response + warning; /health shows history_last_date; the store REFUSES beyond 3 days:")
    print(f"          -> {s['refused']}")
    print("FIX     : run the weekly/nightly refresh (src/models/weekly_job.py); alert when history_last_date < yesterday.")
    facts["stale_history"] = dict(table=t.to_dict("records"), refused=s["refused"])

    print("\n" + "=" * 100 + "\n(b) MAPE EXPLODES ON ZERO-SALES DAYS\n" + "=" * 100)
    z = zero_days(raw)
    print(f"actual [100, 0, 50] vs forecast [90, 10, 55]:  MAPE = {z['tiny']['mape']}   WAPE = {z['tiny']['wape']:.3f}  (=25 units wrong / 150 units sold)")
    print(f"real data: {z['real']['zero_rows']} of {len(raw):,} store-days sold ZERO units (stock-outs) = {z['real']['share']:.2%}.")
    print("SYMPTOM : the weekly MAPE chart jumps to infinity / absurd values after one stock-out.")
    print("CAUSE   : MAPE divides by the actual value of each day; 0 -> division by zero, small values -> huge percentages.")
    print("DETECT  : any metric that is inf/NaN in monitoring; our tests assert WAPE stays finite.")
    print("FIX     : report WAPE (sum of |errors| / total units) and bias; never average per-day percentages.")
    facts["mape_vs_wape"] = z

    print("\n" + "=" * 100 + "\n(c) DUPLICATED DATES AFTER A PIPELINE RETRY\n" + "=" * 100)
    d = duplicated_dates(raw)
    print(f"rolling_mean_7 for the last day of P04/S2:  correct = {d['rolling_right']:.1f}   with the last 3 days loaded twice = {d['rolling_wrong']:.1f}")
    print("SYMPTOM : forecasts drift slightly and nobody knows why - every number still looks plausible.")
    print("CAUSE   : the retry appended rows that already existed; the 7-row window now spans fewer real days.")
    print(f"DETECT  : check_history() raises: {d['check_history']}")
    print(f"          data_quality.validate() fails: {d['dq_failed']}")
    print("FIX     : idempotent loads (upsert_history in weekly_job.py) + this data test in CI and in the retrain ladder.")
    facts["duplicates"] = d

    print("\n" + "=" * 100 + "\n(d) PRICE UNIT CHANGE (rupees -> paise, x100)\n" + "=" * 100)
    p = price_unit_change(raw, art, params)
    print("data quality on the week with prices x100:")
    for x in p["dq_failed"]:
        print("   FAIL", x[:150])
    print(f"\nwhat the first row after the change would do (P03/S1, actual sales {p['actual']:.0f}; price_change becomes +99 instead of ~0):")
    for k, (ok, bad) in p["models"].items():
        print(f"   {k:<16} normal forecast {ok:7.1f} -> with unit error {bad:9.1f}")
    print("SYMPTOM : forecasts for a few products collapse (linear model) or look odd; price drift alarm in the KS table.")
    print("CAUSE   : a source system changed its unit; the code is still 'correct'.")
    print("DETECT  : data-quality check price_unit (median ratio vs reference) + API field range (price <= 5000 -> HTTP 422) + KS drift on price.")
    print("FIX     : fix/convert at the source or in the ingestion step, add a unit test; do NOT retrain on the bad week.")
    facts["price_unit"] = dict(dq=p["dq_failed"], models=p["models"], actual=p["actual"])
    save_fact("failure_scenarios", facts)


if __name__ == "__main__":
    main()
