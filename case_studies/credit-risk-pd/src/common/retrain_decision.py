"""Drift is a SIGNAL, not a command.  Retraining is decided by combining several pieces of evidence.

Decision ladder (cheapest check first):
  1. Data quality   - are nulls / ranges / schema broken?  If yes, FIX THE PIPELINE, do not retrain.
  2. Drift          - did the inputs change (KS test)?
  3. Performance    - did labelled production performance really fall?  (labels often arrive late)
  4. Business       - is the impact big enough to justify cost/risk of a new model?
Only 'RETRAIN_CANDIDATE' means "train a challenger"; it still has to beat the champion before promotion.
"""
from __future__ import annotations

import pandas as pd


def decide(drift_report: pd.DataFrame, data_quality_ok: bool = True, perf_drop: float | None = None,
           perf_drop_threshold: float = 0.03, labels_available: bool = False,
           business_impact_high: bool | None = None) -> dict:
    reasons: list[str] = []
    if not data_quality_ok:
        return dict(decision="FIX_DATA_PIPELINE", reasons=["Data quality checks failed: fix the source/pipeline first. "
                                                          "Retraining on broken data would make things worse."])
    drifted = drift_report.loc[drift_report.status == "drift", "feature"].tolist() if len(drift_report) else []
    small = drift_report.loc[drift_report.status == "significant_but_small", "feature"].tolist() if len(drift_report) else []
    if drifted:
        reasons.append(f"Drift (significant AND large) in: {drifted}.")
    if small:
        reasons.append(f"Statistically significant but small shift in: {small} (watch, do not act alone).")

    if labels_available and perf_drop is not None:
        if perf_drop >= perf_drop_threshold:
            reasons.append(f"Labelled performance fell by {perf_drop:.3f} (>= {perf_drop_threshold}).")
            if business_impact_high is False:
                return dict(decision="MONITOR", reasons=reasons + ["Business impact judged low: keep monitoring."])
            return dict(decision="RETRAIN_CANDIDATE", reasons=reasons + ["Train a challenger; promote only if it beats the champion."])
        reasons.append(f"Labelled performance is stable (drop {perf_drop:.3f} < {perf_drop_threshold}).")
        return dict(decision="MONITOR" if drifted else "NO_ACTION",
                    reasons=reasons + (["Inputs moved but the model still performs: monitor, do not retrain yet."] if drifted else []))

    if drifted:
        return dict(decision="INVESTIGATE",
                    reasons=reasons + ["Labels are not available yet, so performance is unknown. Investigate cause "
                                       "(seasonality? upstream change? real population change?), "
                                       "estimate business impact, and wait for labels before retraining."])
    return dict(decision="NO_ACTION", reasons=reasons or ["No drift evidence and no performance concern."])
