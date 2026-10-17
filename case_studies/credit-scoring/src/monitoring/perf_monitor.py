"""Labelled performance monitoring - the evidence that matters most, and the slowest to arrive.

In lending the label "defaulted within 12 months" is only known 12 months after disbursal.  So:
  * today:   we only have input drift (KS) -> the ladder says INVESTIGATE
  * +12 months: a labelled sample arrives -> we can measure AUC / KS / calibration on REAL outcomes -> maybe RETRAIN_CANDIDATE

Formulas
  perf_drop         = AUC_reference - AUC_production        (reference = champion's TEST AUC stored in the artifact)
  calibration gap   = mean predicted PD - observed default rate.  Tiny example: model says 10% on average, 15% default -> gap -0.05
                      (the model UNDER-estimates risk: the bank will lose more money than it expected).

Run:  python -m src.monitoring.perf_monitor --prod data/production/batch_drifted.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH
from src.features.build_features import engineer
from src.models.metrics import credit_metrics


def score_batch(batch: pd.DataFrame, art: dict) -> np.ndarray:
    frame = engineer(batch)
    return art["pipeline"].predict_proba(frame[art["input_columns"]])[:, 1]


def labelled_performance(batch: pd.DataFrame, art: dict) -> dict:
    p = score_batch(batch, art)
    y = batch[P.TARGET].to_numpy()
    m = credit_metrics(y, p)
    ref_auc = art["metrics"]["roc_auc"]
    bands = pd.Series([P.risk_level(v) for v in p]).value_counts(normalize=True).reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).round(4).to_dict()
    return dict(metrics=m, reference_auc=ref_auc, perf_drop=round(ref_auc - m["roc_auc"], 4),
                mean_predicted_pd=round(float(p.mean()), 4), observed_default_rate=round(float(y.mean()), 4),
                calibration_gap=round(float(p.mean() - y.mean()), 4), band_shares=bands)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    a = ap.parse_args()
    art = load_artifact(CHAMPION_PATH)
    r = labelled_performance(pd.read_csv(a.prod), art)
    print(f"champion {art['version']} on labelled production batch {a.prod}")
    for k, v in r.items():
        print(f"  {k}: {v}")
