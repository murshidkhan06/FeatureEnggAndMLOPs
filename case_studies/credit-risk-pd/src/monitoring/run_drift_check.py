"""Monthly drift check + retraining DECISION (what a scheduled job would run after the monthly scoring batch).

  python -m src.monitoring.run_drift_check --prod data/production/batch_drifted.csv [--today 2024-05-15] [--scheduled]

Steps (cheapest first - the DECISION LADDER):
  1. DATA QUALITY   schema, ranges, null rates vs the reference            -> failure means FIX THE PIPELINE
  2. DRIFT          two-sample KS test per feature: H0 "same distribution", alpha 0.05, plus the effect size D (>= 0.10 = large)
  3. SCORE SHIFT    mean predicted PD and share above the contact threshold  (always available)
  4. PSEUDO-LABEL   early-delinquency rate (needs 3 published months after as_of; depends on --today)
  5. LABELLED       real AUC / calibration (needs 6 months after as_of; depends on --today)
  6. POLICY         src/monitoring/credit_policy.py -> FIX_DATA_PIPELINE | RETRAIN_CANDIDATE | INVESTIGATE | MONITOR | NO_ACTION

--today simulates the calendar: with an early date the labels are NOT known yet (the label-lag problem); with a late date they are.
Nothing is retrained here.  RETRAIN_CANDIDATE only means: run `python -m src.models.retrain` (which still needs a human approval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.config import load_params
from src.common.facts import save_fact
from src.common.drift import explain_row, ks_drift_report
from src.common.paths import CHAMPION_PATH, DATA_RAW, REFERENCE_PATH, REPORTS, SNAPSHOTS_PATH
from src.features.build_features import add_derived_features
from src.features.definitions import TARGET
from src.monitoring.credit_policy import decide_credit_risk
from src.monitoring.data_quality import validate_batch
from src.monitoring.perf_monitor import labelled_performance, score_frame
from src.monitoring.proxy_metrics import early_delinquency_rate, label_available_on, labels_mature, score_shift


def reference_early_delinquency(loan_status: pd.DataFrame, snap: pd.DataFrame, today) -> float | None:
    """The NORMAL early-delinquency rate: average over the validation-period snapshots (calm months)."""
    rates = []
    for as_of, g in snap[snap["split"] == "val"].groupby("as_of_date"):
        r = early_delinquency_rate(loan_status, g["customer_id"].tolist(), as_of, today)
        if r is not None:
            rates.append(r)
    return float(sum(rates) / len(rates)) if rates else None


def run(prod_path: str, ref_path: str, today: str, scheduled: bool, alpha: float, min_effect: float, quiet: bool = False) -> dict:
    params = load_params()
    dp = params["drift"]
    ref, prod = pd.read_csv(ref_path), pd.read_csv(prod_path, parse_dates=["features_as_of"])
    as_of = prod["features_as_of"].iloc[0]
    out = lambda *a, **k: None if quiet else print(*a, **k)  # noqa: E731

    out(f"BATCH  {prod_path}   as_of={as_of.date()}   customers={len(prod):,}   TODAY={today}")
    # 1) data quality -------------------------------------------------------------------------------------------
    dq = validate_batch(prod, ref, dp["max_null_rate_increase"])
    out("\n[1] DATA QUALITY:", "OK" if dq["ok"] else "FAILED")
    if not quiet:
        print(dq["table"].to_string(index=False))
        for p in dq["problems"]:
            print("   !", p)
    # 2) drift --------------------------------------------------------------------------------------------------
    rep = ks_drift_report(ref, prod, [f for f in P.DRIFT_FEATURES if f in prod.columns], alpha=alpha, min_effect=min_effect)
    if not quiet:
        print(f"\n[2] DRIFT (two-sample KS test).  H0: reference and production come from the SAME distribution.  alpha={alpha}, "
              f"'large' means D >= {min_effect}")
        show = rep.copy()
        refm, prodm = ref[rep["feature"]].mean(), pd.to_numeric(prod[rep["feature"]].stack().groupby(level=1).mean())
        show.insert(1, "ref_mean", [round(float(refm[f]), 3) for f in rep["feature"]])
        show.insert(2, "prod_mean", [round(float(prodm.get(f, float("nan"))), 3) for f in rep["feature"]])
        show["p_value"] = show["p_value"].map(lambda v: f"{v:.2e}")
        print(show.drop(columns=["n_ref", "n_prod", "statistically_significant", "practically_significant"]).to_string(index=False))
        print()
        for _, r in rep.iterrows():
            print(" -", explain_row(r, alpha))
    # 3) score shift --------------------------------------------------------------------------------------------
    labelled = pseudo = None
    sh = None
    if CHAMPION_PATH.exists():
        art = load_artifact(CHAMPION_PATH)
        scores = score_frame(art, prod)
        sh = score_shift(art, scores, float(art["extra"].get("test_mean_pd", art["extra"]["val_metrics"]["mean_score"])), params["model"]["top_pct"])
        out(f"\n[3] SCORE SHIFT (champion {art['version']}): mean PD {sh['mean_pd']:.3f} vs {sh['ref_mean_pd']:.3f} reference (x{sh['mean_pd_ratio']}); "
            f"{sh['contact_rate']:.1%} of customers above the contact threshold vs {sh['planned_contact_rate']:.0%} planned (x{sh['contact_rate_ratio']})")
        # 4) pseudo label ---------------------------------------------------------------------------------------
        ls_path = DATA_RAW / "loan_status.csv"
        if ls_path.exists() and SNAPSHOTS_PATH.exists() and "customer_id" in prod.columns:
            ls = pd.read_csv(ls_path, parse_dates=["month", "available_at"])
            snap = pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"])
            cur = early_delinquency_rate(ls, prod["customer_id"].tolist(), as_of, today)
            base = reference_early_delinquency(ls, snap, today)
            if cur is None:
                out(f"\n[4] PSEUDO-LABEL (early delinquency in the 3 months after as_of): NOT READY - needs loan_status published through "
                    f"{(as_of + pd.offsets.MonthEnd(3) + pd.Timedelta(days=5)).date()}")
            else:
                pseudo = {"rate": round(cur, 4), "normal_rate": round(base, 4), "rate_ratio": round(cur / max(base, 1e-9), 2)}
                out(f"\n[4] PSEUDO-LABEL: {cur:.1%} of customers missed >= 1 instalment in the 3 months after as_of "
                    f"(normal {base:.1%}) -> x{pseudo['rate_ratio']}")
        # 5) labelled -------------------------------------------------------------------------------------------
        if TARGET in prod.columns and labels_mature(as_of, today):
            labelled = labelled_performance(art, prod, params["model"]["top_pct"])
            m = labelled["metrics"]
            out(f"\n[5] LABELLED PERFORMANCE (labels known since {label_available_on(as_of).date()}): ROC-AUC {m['roc_auc']} "
                f"(reference {art['metrics']['roc_auc']}, drop {labelled['auc_drop']:+.3f}); recall@10% {m['recall_at_10pct']} "
                f"(reference {art['metrics']['recall_at_10pct']}); mean PD {labelled['mean_pd']} vs observed default rate {labelled['observed_rate']} "
                f"(x{labelled['calibration_ratio']})")
        else:
            out(f"\n[5] LABELLED PERFORMANCE: NOT AVAILABLE - the 6-month label of {as_of.date()} is known only on {label_available_on(as_of).date()} "
                f"(today is {today}). Performance is UNKNOWN; do not pretend otherwise.")
    # 6) policy -------------------------------------------------------------------------------------------------
    decision = decide_credit_risk(rep, dp["key_features"], dq["ok"], dq["problems"], labelled, pseudo, scheduled,
                                  dp["min_key_features"], dp["perf_drop_threshold"], dp["calibration_ratio"], 1.5)
    out("\n[6] DECISION:", decision["decision"])
    for r in decision["reasons"]:
        out("   *", r)
    result = dict(batch=str(prod_path), as_of=str(as_of.date()), today=str(today), data_quality=dict(ok=dq["ok"], problems=dq["problems"]),
                  drift=rep.to_dict("records"), score_shift=sh, pseudo_label=pseudo,
                  labelled=None if labelled is None else {k: v for k, v in labelled.items()}, decision=decision)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift_report.json").write_text(json.dumps(result, indent=2, default=str))
    save_fact(f"drift_{Path(prod_path).stem}_{today}", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--ref", default=str(REFERENCE_PATH))
    ap.add_argument("--today", default=load_params()["retrain"]["today"], help="the 'current date' used to decide which labels exist")
    ap.add_argument("--scheduled", action="store_true", help="a scheduled quarterly retrain is due")
    ap.add_argument("--alpha", type=float, default=load_params()["drift"]["alpha"])
    ap.add_argument("--min-effect", type=float, default=load_params()["drift"]["min_effect"])
    a = ap.parse_args()
    run(a.prod, a.ref, a.today, a.scheduled, a.alpha, a.min_effect)
