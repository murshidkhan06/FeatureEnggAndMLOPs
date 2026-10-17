"""Labelled performance monitoring: do the forecasts still match what customers actually bought?

Forecasting has a gift most ML problems lack: LABELS ARRIVE FAST.  Tomorrow we know today's forecast error.  So we can monitor
the real error (WAPE, bias) every week - not only the input distributions.

TRIGGER RULE (configs/retrain_policy.yaml)
  weekly WAPE  >=  validation WAPE x (1 + 25%)   in 2 CONSECUTIVE complete weeks     (performance evidence)
  AND the KS test shows real drift in the lag / rolling features                       (input evidence)
Two weeks in a row filters out one bad week (a holiday, one promotion); the lag-drift condition makes sure the cause looks like
a genuine change of demand level and not a broken data feed.

Worked example: validation WAPE 0.130 -> alarm level 0.130 x 1.25 = 0.1625.
  week 5 WAPE 0.171, week 6 WAPE 0.168 -> both above 0.1625 -> 2 consecutive breaches -> performance evidence = YES.

Run:  python -m src.monitoring.performance --prod data/production/batch_drifted.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.artifact import load_artifact
from src.common.facts import save_fact
from src.common.params import load_policy
from src.common.paths import CHAMPION_PATH, DATA_RAW
from src.features.build_features import features_for_batch
from src.models import metrics as M


def labelled_features(batch_path, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Production batch (with TRUE sales) -> feature rows, using training history for the first days' lags."""
    if history is None:
        history = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    b = pd.read_csv(batch_path)
    f = features_for_batch(b, history)
    return f[f["sales"].notna()].reset_index(drop=True)


def weekly_performance(frame: pd.DataFrame, pred: np.ndarray, validation_wape: float, worsen_pct: float) -> pd.DataFrame:
    """One row per COMPLETE 7-day block counted from the first date of the batch."""
    d = frame.assign(pred=pred)
    d["week"] = (pd.to_datetime(d["date"]) - pd.to_datetime(d["date"]).min()).dt.days // 7 + 1
    days = d.groupby("week")["date"].nunique()
    full = days[days == 7].index
    rows = []
    for w, g in d[d.week.isin(full)].groupby("week"):
        wape = M.wape(g.sales, g.pred)
        rows.append(dict(week=int(w), from_date=str(g.date.min().date()), actual_units=int(g.sales.sum()),
                         forecast_units=int(g.pred.sum()), wape=round(wape, 4), bias_pct=round(M.bias_pct(g.sales, g.pred), 4),
                         vs_validation=round(wape / validation_wape - 1, 4)))
    t = pd.DataFrame(rows)
    t["breach"] = t["vs_validation"] >= worsen_pct
    return t


def trigger_status(weekly: pd.DataFrame, consecutive: int) -> dict:
    """perf_drop = the SMALLEST relative worsening among the last `consecutive` weeks (so it is >= threshold only if ALL breach)."""
    last = weekly.tail(consecutive)
    perf_drop = float(last["vs_validation"].min()) if len(last) == consecutive else 0.0
    return dict(perf_drop=perf_drop, breaches_in_a_row=int(weekly["breach"].iloc[::-1].cumprod().sum()),
                last_weeks_bias_pct=float(last["bias_pct"].mean()) if len(last) else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    a = ap.parse_args()
    pol = load_policy()["trigger"]
    art = load_artifact(CHAMPION_PATH)
    f = labelled_features(a.prod)
    from src.models.train import score
    pred = score(art["pipeline"], f, art["input_columns"])
    val = art["metrics"]["val_wape"]
    wk = weekly_performance(f, pred, val, pol["wape_worsen_pct"])
    st = trigger_status(wk, pol["consecutive_weeks"])
    print(f"champion {art['version']} ({art['extra']['model_name']}); validation WAPE = {val:.4f}; "
          f"alarm level = {val * (1 + pol['wape_worsen_pct']):.4f} (+{pol['wape_worsen_pct']:.0%})\n")
    print(wk.to_string(index=False))
    print(f"\nconsecutive breaching weeks at the end: {st['breaches_in_a_row']} (rule needs {pol['consecutive_weeks']})")
    print(f"performance evidence: {'YES' if st['perf_drop'] >= pol['wape_worsen_pct'] else 'NO'}; "
          f"average bias of the last {pol['consecutive_weeks']} weeks = {st['last_weeks_bias_pct']:+.1%}")
    save_fact("performance_" + a.prod.replace("\\", "/").split("/")[-1].replace(".csv", ""), dict(weekly=wk.to_dict("records"), status=st))


if __name__ == "__main__":
    main()
