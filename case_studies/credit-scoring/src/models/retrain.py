"""Retraining workflow with CHAMPION / CHALLENGER promotion and a HUMAN approval step.

   production batch
     -> 1 data-quality validation        broken data?  -> FIX_DATA_PIPELINE (stop; never retrain on a bug)
     -> 2 drift check (KS test)          signal only
     -> 3 decision ladder                labels missing -> INVESTIGATE (stop).  Labels + performance drop -> RETRAIN_CANDIDATE
     -> 4 train a CHALLENGER             on refreshed data (old training data + the older 60% of the labelled batch, fresh rows weighted x3)
     -> 5 evaluate champion AND challenger on the SAME newer hold-out (the newest 40% of the batch, never used for training)
     -> 6 promotion rules + guardrails   challenger AUC >= champion AUC + 0.005, KS / recall may not fall > 0.03, fairness AIR >= 0.80,
                                         no collapse on the old test set
     -> 7 approval record printed        who must sign: model owner, risk, compliance
     -> 8 promote ONLY if every rule passed AND a named human approved (--approve-as NAME)

NEVER "drift detected = automatically retrain".  Drift only starts the investigation.

Usage
  python -m src.models.retrain --prod data/production/batch_drifted.csv                       # no labels yet  -> INVESTIGATE
  python -m src.models.retrain --prod data/production/batch_drifted.csv --labels-available    # train + compare, stops at "awaiting approval"
  python -m src.models.retrain --prod data/production/batch_drifted.csv --labels-available --approve-as "A. Rao (Model Risk)"
  python -m src.models.retrain --rollback                                                      # put the previous champion back
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd

from src import project as P
from src.common import promotion
from src.common.artifact import load_artifact, save_artifact
from src.common.drift import ks_drift_report
from src.common.mlflow_utils import setup, tracked_run
from src.common.params import load_params
from src.common.paths import CHALLENGER_PATH, CHAMPION_PATH, REFERENCE_PATH, REPORTS
from src.common.retrain_decision import decide
from src.features.build_features import engineer
from src.features.registry import check_features_approved
from src.governance.fairness_audit import adverse_impact_ratio, group_table
from src.models.metrics import credit_metrics
from src.models.pipeline import build_pipeline, load_splits
from src.monitoring.data_quality import validate_batch
from src.monitoring.perf_monitor import labelled_performance
from src.monitoring.run_drift_check import business_evidence


def split_batch(raw: pd.DataFrame, holdout_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest (1 - holdout_frac) of the batch = fresh training rows; newest holdout_frac = hold-out (time order, no shuffling)."""
    d = raw.sort_values(["application_date", "applicant_id"], kind="stable").reset_index(drop=True)
    cut = int(len(d) * (1 - holdout_frac))
    return d.iloc[:cut], d.iloc[cut:]


def holdout_air(df_raw: pd.DataFrame, p: np.ndarray, cutoff: float) -> float:
    t = group_table(df_raw["gender"], df_raw[P.TARGET], p >= cutoff)
    return adverse_impact_ratio(t.approval_rate)


def run_retraining(prod_path: str, labels_available: bool, approve_as: str | None, version: str) -> dict:
    prm = load_params()
    rt = prm["retrain"]
    art = load_artifact(CHAMPION_PATH)
    feats = art["input_columns"]
    ref, raw = pd.read_csv(REFERENCE_PATH), pd.read_csv(prod_path)
    print(f"CHAMPION: {art['version']} (test AUC {art['metrics']['roc_auc']})   batch: {prod_path} ({len(raw)} applications)\n")

    # ---- 1-3  evidence + decision ladder ------------------------------------------------------------------------------
    dq = validate_batch(raw)
    print("STEP 1  data quality:", "OK" if dq["ok"] else f"FAILED {dq['problems']}")
    frame = engineer(raw)
    rep = ks_drift_report(ref, frame, P.DRIFT_FEATURES)
    print("STEP 2  drift (KS test):", {r.feature: f"D={r.ks_statistic:.2f} {r.status}" for r in rep.itertuples()})
    labelled = labelled_performance(raw, art) if labels_available else None
    perf_drop = labelled["perf_drop"] if labelled else None
    if labelled:
        print(f"STEP 3  labelled performance: AUC {labelled['metrics']['roc_auc']} vs {labelled['reference_auc']} -> drop {perf_drop}; "
              f"predicted PD {labelled['mean_predicted_pd']} vs observed {labelled['observed_default_rate']}")
    else:
        print("STEP 3  labelled performance: labels not available yet (defaults are known 12 months after disbursal)")
    biz, impact = business_evidence(raw, art, labelled)
    d = decide(rep, data_quality_ok=dq["ok"], perf_drop=perf_drop, perf_drop_threshold=rt["perf_drop_threshold"],
               labels_available=labelled is not None, business_impact_high=impact if labelled else None)
    print(f"\nDECISION LADDER -> {d['decision']}")
    for r in d["reasons"]:
        print("   *", r)
    if d["decision"] != "RETRAIN_CANDIDATE":
        print(f"\nNo challenger is trained: the ladder says {d['decision']}.  (Drift alone never triggers retraining.)")
        return dict(decision=d["decision"], promoted=False)

    # ---- 4  challenger on refreshed data ----------------------------------------------------------------------------------
    fresh, hold = split_batch(raw, rt["holdout_frac"])
    assert not set(fresh.applicant_id) & set(hold.applicant_id), "applicant overlap between fresh training rows and hold-out"
    train, valid, _ = load_splits()
    old = pd.concat([train, valid], ignore_index=True)
    fresh_f, hold_f = engineer(fresh), engineer(hold)
    X = pd.concat([old[feats], fresh_f[feats]], ignore_index=True)
    y = pd.concat([old[P.TARGET], fresh_f[P.TARGET]], ignore_index=True)
    w = np.r_[np.ones(len(old)), np.full(len(fresh_f), rt["recency_weight"])]
    check_features_approved(feats)
    print(f"\nSTEP 4  train CHALLENGER {version}: {len(old)} old rows + {len(fresh_f)} fresh rows (weight x{rt['recency_weight']}), "
          f"hold-out = newest {len(hold_f)} rows ({hold.application_date.min()} .. {hold.application_date.max()}), same features as champion")
    setup(P.MODEL_NAME)
    with tracked_run(f"challenger_{version}", dict(model="logreg", C=prm["model"]["C"], version=version, n_old=len(old), n_fresh=len(fresh_f),
                                                   recency_weight=rt["recency_weight"]), tags={"stage": "retraining"}):
        pipe = build_pipeline(feats, "logreg", prm["model"]["C"]).fit(X, y, clf__sample_weight=w)
        p_ch, p_cp = pipe.predict_proba(hold_f[feats])[:, 1], art["pipeline"].predict_proba(hold_f[feats])[:, 1]
        m_ch, m_cp = credit_metrics(hold_f[P.TARGET], p_ch), credit_metrics(hold_f[P.TARGET], p_cp)
        for k, v in m_ch.items():
            if isinstance(v, float):
                mlflow.log_metric(f"holdout_{k}", v)

    # ---- 5-6  same hold-out, promotion rules -------------------------------------------------------------------------------
    old_test = load_splits()[2]
    old_auc_ch = credit_metrics(old_test[P.TARGET], pipe.predict_proba(old_test[feats])[:, 1])["roc_auc"]
    old_auc_cp = art["metrics"]["roc_auc"]
    air_ch, air_cp = holdout_air(hold, p_ch, art["threshold"]), holdout_air(hold, p_cp, art["threshold"])
    cmp = promotion.compare(m_cp, m_ch, primary="roc_auc", min_improvement=rt["min_improvement"],
                            guardrails={"ks_stat": rt["guardrail_ks_drop"], "recall_at_80_approval": rt["guardrail_recall_drop"]})
    extra_checks = {f"fairness_AIR>={prm['fairness']['min_adverse_impact_ratio']}": bool(air_ch >= prm["fairness"]["min_adverse_impact_ratio"]),
                    "old_test_AUC_drop<=0.02": bool(old_auc_cp - old_auc_ch <= 0.02)}
    checks = {**cmp["checks"], **extra_checks}
    promote_ok = all(checks.values())
    print("\nSTEP 5  champion vs challenger on the SAME newer hold-out")
    print(pd.DataFrame({"champion": m_cp, "challenger": m_ch}).loc[["roc_auc", "gini", "ks_stat", "pr_auc", "recall_at_80_approval", "brier"]].to_string())
    print(f"   gender AIR on hold-out: champion {air_cp:.3f} | challenger {air_ch:.3f};   AUC on the OLD test set: champion {old_auc_cp} | challenger {old_auc_ch}")
    print("\nSTEP 6  promotion rules (all must be True)")
    for k, v in checks.items():
        print(f"   [{'PASS' if v else 'FAIL'}] {k}")
    print(f"   AUC improvement on hold-out: {cmp['delta']:+.4f}")

    # ---- 7  approval record -------------------------------------------------------------------------------------------------
    save_artifact(CHALLENGER_PATH, pipe, version, feats, {k: v for k, v in m_ch.items() if k not in ("n", "default_rate")},
                  threshold=art["threshold"], extra=dict(model_family=art["extra"].get("model_family"), features=feats, trained_on="old + fresh (weighted)",
                                                         holdout_metrics=m_ch, champion_holdout_metrics=m_cp))
    record = dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"), champion=art["version"], challenger=version,
                  reason=d["reasons"], checks=checks, rules_pass=promote_ok, holdout_delta_auc=cmp["delta"],
                  required_signoffs=["Model owner", "Model risk / independent validation", "Compliance (fairness)"], approved_by=approve_as)
    print("\nSTEP 7  APPROVAL RECORD")
    print(json.dumps(record, indent=2))
    result = dict(decision=d["decision"], promoted=False, rules_pass=promote_ok, delta_auc=cmp["delta"], champion_holdout=m_cp, challenger_holdout=m_ch,
                  checks=checks, challenger_version=version, air_champion=round(air_cp, 4), air_challenger=round(air_ch, 4))
    if not promote_ok:
        print("\nNOT PROMOTED: at least one rule failed.  The champion keeps serving.")
    elif not approve_as:
        print("\nAWAITING HUMAN APPROVAL: every rule passed, but a named person must sign off.  Re-run with  --approve-as \"NAME (ROLE)\".")
    else:
        newv = promotion.promote_challenger(approve_as, f"AUC +{cmp['delta']} on newer hold-out; all guardrails passed")
        print(f"\nPROMOTED: {newv} is now the champion (approved by {approve_as}).  Previous champion archived as models/model_{art['version']}.joblib."
              "\nRestart / reload the API (or roll the Kubernetes deployment) to serve it.  Roll back with:  python -m src.models.retrain --rollback")
        result["promoted"] = True
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retrain_record.json").write_text(json.dumps(dict(record=record, result=result), indent=2, default=str))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default="data/production/batch_drifted.csv")
    ap.add_argument("--labels-available", action="store_true", help="the 12-month labels of the batch have arrived")
    ap.add_argument("--approve-as", default=None, help="name + role of the human approver (simulates the sign-off)")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()
    if a.rollback:
        prev = promotion.rollback()
        print(f"ROLLED BACK: champion is now {prev}" if prev else "nothing to roll back to (no archived model in the registry)")
        return
    run_retraining(a.prod, a.labels_available, a.approve_as, a.version)


if __name__ == "__main__":
    main()
