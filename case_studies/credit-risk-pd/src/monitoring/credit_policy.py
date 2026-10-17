"""The credit-risk RETRAINING POLICY: drift is a SIGNAL, never a command.

It wraps the shared decision ladder (src/common/retrain_decision.py) with the bank's own rules:

  step 0  DATA QUALITY first.        Broken inputs -> FIX_DATA_PIPELINE (never retrain on broken data).
  step 1  KEY-FEATURE DRIFT.         Count the key features whose KS result is 'drift' (significant AND D >= 0.10).
  step 2  OUTCOME EVIDENCE.          at least one of
            a) labelled performance fell: ROC-AUC drop >= 0.03  OR  calibration off by >= 1.5x      (needs 6-month labels)
            b) pseudo-label evidence: early-delinquency rate >= 1.5x the normal rate                  (needs ~3 months)
  step 3  RULE   RETRAIN_CANDIDATE  if  (drift in >= 2 key features  AND  outcome evidence)  OR  a scheduled quarterly retrain is due.
          'RETRAIN_CANDIDATE' only means "train a challenger".  The challenger must still beat the champion on a newer hold-out,
          pass the guardrails and receive a human approval before anything is promoted.

OUTCOMES:  FIX_DATA_PIPELINE | RETRAIN_CANDIDATE | INVESTIGATE | MONITOR | NO_ACTION

TINY EXAMPLE
  4 key features drift, labels not yet available, early-delinquency ratio 2.1  -> RETRAIN_CANDIDATE (pseudo-label evidence)
  4 key features drift, labels not yet available, no pseudo-label yet          -> INVESTIGATE (wait for evidence)
  1 key feature drifts + AUC fell 0.05                                         -> INVESTIGATE (not broad: look for another cause)
  account_age shifts with D = 0.06 (significant but small)                     -> MONITOR, nothing else happens
"""
from __future__ import annotations

import pandas as pd

from src.common.retrain_decision import decide


def decide_credit_risk(drift_report: pd.DataFrame, key_features: list[str], dq_ok: bool, dq_problems: list[str] | None = None,
                       labelled: dict | None = None, pseudo: dict | None = None, scheduled_due: bool = False,
                       min_key_features: int = 2, perf_drop_threshold: float = 0.03, calibration_ratio: float = 1.5,
                       pseudo_ratio: float = 1.5) -> dict:
    """labelled = {'auc_drop','calibration_ratio','contact_rate_ratio','recall_drop'} or None;  pseudo = {'rate_ratio'} or None."""
    n_key = int(drift_report.loc[drift_report["feature"].isin(key_features) & (drift_report["status"] == "drift"), "feature"].nunique())
    drifted_key = drift_report.loc[drift_report["feature"].isin(key_features) & (drift_report["status"] == "drift"), "feature"].tolist()
    business_high = None
    if labelled is not None:
        business_high = bool(labelled.get("contact_rate_ratio", 0) >= 1.5 or labelled.get("recall_drop", 0) >= 0.05)
    base = decide(drift_report, data_quality_ok=dq_ok, perf_drop=None if labelled is None else labelled["auc_drop"],
                  perf_drop_threshold=perf_drop_threshold, labels_available=labelled is not None, business_impact_high=business_high)
    evidence = {"key_features_drifting": drifted_key, "n_key": n_key, "labelled": labelled, "pseudo": pseudo, "scheduled_due": scheduled_due}
    if not dq_ok:
        return dict(decision="FIX_DATA_PIPELINE", reasons=base["reasons"] + [f"Problems: {dq_problems}"], evidence=evidence)

    reasons = [r for r in base["reasons"] if r.startswith(("Drift", "Statistically"))]     # keep the drift facts, drop the shared ladder's verdict
    reasons.append(f"Key features with significant AND large drift: {n_key} ({drifted_key}); rule needs >= {min_key_features}.")
    ev_lab = labelled is not None and (labelled["auc_drop"] >= perf_drop_threshold or labelled["calibration_ratio"] >= calibration_ratio)
    ev_pseudo = pseudo is not None and pseudo["rate_ratio"] >= pseudo_ratio
    if labelled is not None:
        reasons.append(f"Labelled evidence: AUC drop {labelled['auc_drop']:+.3f} (threshold {perf_drop_threshold}), calibration x{labelled['calibration_ratio']:.2f}, "
                       f"contact rate x{labelled['contact_rate_ratio']:.2f} of plan -> {'EVIDENCE' if ev_lab else 'no evidence'}.")
    else:
        reasons.append("Labelled evidence: NOT AVAILABLE yet (labels arrive 6 months after as_of).")
    if pseudo is not None:
        reasons.append(f"Pseudo-label (early delinquency): x{pseudo['rate_ratio']:.2f} of the normal rate -> {'EVIDENCE' if ev_pseudo else 'no evidence'}.")
    else:
        reasons.append("Pseudo-label (early delinquency): NOT READY yet (needs 3 published months).")

    if n_key >= min_key_features and (ev_lab or ev_pseudo):
        return dict(decision="RETRAIN_CANDIDATE", evidence=evidence,
                    reasons=reasons + ["Rule met: broad key-feature drift AND outcome evidence. Train a challenger; promote only if it beats the champion and a human approves."])
    if scheduled_due:
        return dict(decision="RETRAIN_CANDIDATE", evidence=evidence,
                    reasons=reasons + ["Scheduled quarterly retrain is due: train a challenger regardless of drift (it still has to win and be approved)."])
    if n_key >= min_key_features:
        return dict(decision="INVESTIGATE", evidence=evidence,
                    reasons=reasons + ["Broad drift but no outcome evidence yet: find the cause (macro shock? upstream change?), estimate the business impact, "
                                       "watch the proxy metrics and wait for labels. Do NOT retrain on drift alone."])
    if ev_lab:
        return dict(decision="INVESTIGATE", evidence=evidence,
                    reasons=reasons + ["Performance fell but the drift is not broad: look for concept change or a data problem before retraining."])
    anything = bool((drift_report["status"].isin(["drift", "significant_but_small"])).any())
    return dict(decision="MONITOR" if anything else "NO_ACTION", evidence=evidence,
                reasons=reasons + (["Some shifts exist (see table) but none justifies action: keep watching."] if anything else ["Nothing to do."]))
