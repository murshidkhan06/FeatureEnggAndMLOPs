"""LABELLED performance monitoring - and why it always arrives LATE.

For month-end `as_of` the true label is known only on  as_of + 6 months.  This module (a) turns a labelled batch into the
numbers the policy needs and (b) prints the "label lag" timeline: for any 'today', which months can already be graded?

  auc_drop            reference ROC-AUC (champion's test period) minus the ROC-AUC on this batch
  calibration_ratio   max(mean predicted PD / observed default rate, observed / predicted)   (1.0 = perfectly calibrated on average)
  contact_rate_ratio  share of customers above the FIXED contact threshold / planned 10 %   (2.0 = collections gets twice the calls)
  recall_drop         reference recall@10 % minus recall@10 % on this batch

TINY EXAMPLE
  champion test AUC 0.885, batch AUC 0.798 -> auc_drop 0.087.   mean PD 8.6 %, observed default rate 7.0 % -> calibration x1.23.

CLI:  python -m src.monitoring.perf_monitor --today 2024-12-31        (month-by-month table; blank where labels are not mature)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.artifact import load_artifact
from src.common.config import load_params
from src.common.paths import CHAMPION_PATH, SNAPSHOTS_PATH
from src.features.build_features import add_derived_features
from src.features.definitions import TARGET
from src.models.evaluate import compute_metrics
from src.monitoring.proxy_metrics import label_available_on, labels_mature


def score_frame(art: dict, df: pd.DataFrame) -> np.ndarray:
    """Score a frame of base features with a model artifact (derived features are added exactly as in training/serving)."""
    return art["pipeline"].predict_proba(add_derived_features(df)[art["input_columns"]])[:, 1]


def labelled_performance(art: dict, df: pd.DataFrame, top_pct: float = 0.10) -> dict:
    """Compare a LABELLED batch with the champion's reference numbers."""
    score = score_frame(art, df)
    m = compute_metrics(df[TARGET], score, top_pct)
    pred, obs = float(np.mean(score)), float(df[TARGET].mean())
    return {"metrics": m, "auc_drop": round(art["metrics"]["roc_auc"] - m["roc_auc"], 4),
            "recall_drop": round(art["metrics"]["recall_at_10pct"] - m["recall_at_10pct"], 4),
            "calibration_ratio": round(max(pred / max(obs, 1e-9), obs / max(pred, 1e-9)), 3),
            "mean_pd": round(pred, 4), "observed_rate": round(obs, 4),
            "contact_rate_ratio": round(float((score >= art["threshold"]).mean()) / top_pct, 3)}


def timeline(art: dict, snap: pd.DataFrame, today, top_pct: float = 0.10) -> pd.DataFrame:
    """One row per production month: scored customers, mean PD, label maturity and (if mature) the real performance."""
    rows = []
    for as_of, g in snap[snap["split"] == "production"].groupby("as_of_date"):
        score = score_frame(art, g)
        mature = labels_mature(as_of, today)
        r = dict(as_of=as_of.date(), customers=len(g), mean_pd=round(float(score.mean()), 4),
                 contact_rate=round(float((score >= art["threshold"]).mean()), 3),
                 labels_known_on=label_available_on(as_of).date(), labelled="YES" if mature else "not yet")
        if mature:
            m = compute_metrics(g[TARGET], score, top_pct)
            r.update(observed_default_rate=m["base_rate"], roc_auc=m["roc_auc"], recall_at_10pct=m["recall_at_10pct"])
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=load_params()["retrain"]["today"])
    a = ap.parse_args()
    art = load_artifact(CHAMPION_PATH)
    snap = add_derived_features(pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"]))
    tl = timeline(art, snap, a.today)
    pd.set_option("display.width", 220)
    print(f"Champion {art['version']}: reference test ROC-AUC {art['metrics']['roc_auc']}, recall@10% {art['metrics']['recall_at_10pct']}, "
          f"contact threshold {art['threshold']:.4f}")
    print(f"TODAY = {a.today}.  A month can be graded only once its 6-month label is known (label_known_on <= today).\n")
    print(tl.fillna("").to_string(index=False))
    n_un = int((tl["labelled"] == "not yet").sum())
    from src.common.facts import save_fact
    save_fact(f"label_lag_{a.today}", dict(today=a.today, timeline=tl.to_dict("records")))
    print(f"\n{n_un} of {len(tl)} months cannot be graded yet: for those, only proxy evidence (drift, score shift, early delinquency) exists.")
