"""RETRAINING WORKFLOW with a human in the loop.  Drift is a SIGNAL; retraining is a DECISION; promotion needs a SIGN-OFF.

  python -m src.models.retrain --prod data/production/batch_drifted.csv                       stops at "awaiting approval"
  python -m src.models.retrain --prod data/production/batch_drifted.csv --approve-as "Priya (Model Risk)"   promotes if the rules pass
  python -m src.models.retrain --rollback                                                      put the previous champion back
  python -m src.models.retrain --prod ... --window stress_only --train-anyway                  FAILURE SCENARIO (c), see below

THE EIGHT STEPS (each prints a heading, so a student can follow the log)
  1. DATA QUALITY    validate the production batch.  Broken data -> stop (FIX_DATA_PIPELINE).
  2. DRIFT           two-sample KS test per feature (H0: same distribution, alpha 0.05, large = D >= 0.10).
  3. DECISION LADDER src/monitoring/credit_policy.py.  Only RETRAIN_CANDIDATE continues (drift alone never does).
  4. TRAIN CHALLENGER on the refreshed window (labels must be KNOWN by `today`: as_of + 6 months <= today).
  5. COMPARE both models on the SAME newer hold-out (a period neither of them was trained on).
  6. PROMOTION RULES promotion.compare: ROC-AUC must improve by >= min_improvement AND the guardrail (recall@10 %) may not fall.
  7. APPROVAL RECORD printed + saved (reports/approval_record.json).  No name -> "AWAITING APPROVAL", nothing changes.
  8. PROMOTE only if the rules pass AND a human name was given (--approve-as).  The old champion is archived -> rollback possible.

TRAINING WINDOWS (--window)
  rolling       every labelled snapshot up to retrain.train_end (calm months + stress months)           <- the sensible default
  stress_only   ONLY snapshots from retrain.stress_window_start on (the "recent behaviour" shortcut)
  FAILURE SCENARIO (c): a challenger trained on the stress period only has seen few months, one macro regime and few defaulters.
  It is then worse on calm (normal-economy) data and has no better view of the future.  A window must cover the regimes the model
  will meet.  `--calm-check` prints that comparison.

LABEL MATURITY  A snapshot is usable for training only if its 6-month label is known by `today`:
  as_of + 6 months <= today.   Example: today = 2025-06-30 -> the last usable as_of is 2024-12-31.
  And we leave a PURGE gap between the training window and the hold-out (train_end 2024-04-30, hold-out starts 2024-10-31), because
  customers appear in many snapshots and the label windows of neighbouring months overlap.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

import pandas as pd

from src.common import promotion
from src.common.artifact import load_artifact, save_artifact
from src.common.config import load_params
from src.common.facts import save_fact
from src.common.paths import CHALLENGER_PATH, CHAMPION_PATH, DATA_PROCESSED, REPORTS, REGISTRY_PATH
from src.features.definitions import TARGET
from src.models.evaluate import business_summary, compute_metrics, evaluate_pipeline
from src.models.train import fit_model, load_snapshots
from src.monitoring.run_drift_check import run as run_drift_check
from src.monitoring.proxy_metrics import label_available_on

THRESHOLD_MONTHS = ("2024-07-31", "2024-09-30")      # unlabelled recent months: only used to set the contact threshold (needs no labels)


def mature(snap: pd.DataFrame, today) -> pd.DataFrame:
    """Keep rows whose 6-month label is known by `today`.   Example: today 2025-06-30 keeps as_of <= 2024-12-31."""
    ok = snap["as_of_date"].map(lambda d: label_available_on(d) <= pd.Timestamp(today))
    return snap[ok]


def training_window(snap: pd.DataFrame, rp: dict, window: str) -> pd.DataFrame:
    """Rows the challenger may learn from."""
    d = snap[snap["as_of_date"] <= pd.Timestamp(rp["train_end"])]
    if window == "stress_only":
        d = d[d["as_of_date"] >= pd.Timestamp(rp["stress_window_start"])]
    return mature(d, rp["today"])


def _months_between(a, b) -> int:
    """Whole months from date a to date b.  ('2024-04-30', '2024-10-31') -> 6."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month)


def next_version(champion_version: str) -> str:
    """'v1' -> 'v2'.  (Any other pattern gets '-next' appended.)"""
    m = re.fullmatch(r"v(\d+)", champion_version)
    return f"v{int(m.group(1)) + 1}" if m else champion_version + "-next"


def train_challenger(train: pd.DataFrame, snap: pd.DataFrame, champion: dict, version: str, rp: dict, hold: pd.DataFrame,
                     top_pct: float, params: dict) -> dict:
    """Fit a challenger of the SAME family and features as the champion on `train`; return its artifact fields.

    Contact threshold = the (1 - top_pct) quantile of the challenger's scores in recent months.  That needs NO labels, so
    we may use the newest months even though their outcome is unknown."""
    kind, feats = champion["extra"]["kind"], champion["input_columns"]
    pipe = fit_model(train, feats, kind, params)
    thr_rows = snap[(snap["as_of_date"] >= pd.Timestamp(THRESHOLD_MONTHS[0])) & (snap["as_of_date"] <= pd.Timestamp(THRESHOLD_MONTHS[1]))]
    threshold = float(pd.Series(pipe.predict_proba(thr_rows[feats])[:, 1]).quantile(1 - top_pct))
    m_hold, s_hold = evaluate_pipeline(pipe, hold, feats, TARGET, top_pct)
    return dict(pipe=pipe, kind=kind, feats=feats, threshold=threshold, metrics=m_hold, scores=s_hold)


def _print_compare(name_a: str, ma: dict, name_b: str, mb: dict) -> None:
    keys = ("roc_auc", "pr_auc", "recall_at_10pct", "lift_at_10pct", "ks")
    print(f"   {'metric':<18}{name_a:>14}{name_b:>14}{'difference':>14}")
    for k in keys:
        print(f"   {k:<18}{ma[k]:>14.4f}{mb[k]:>14.4f}{mb[k] - ma[k]:>+14.4f}")


def calm_check(champion: dict, train_rolling: pd.DataFrame, train_stress: pd.DataFrame, snap: pd.DataFrame, params: dict, top: float) -> dict:
    """FAILURE SCENARIO (c): rolling-window challenger vs stress-only challenger on STRESS hold-out and on CALM data."""
    feats, kind = champion["input_columns"], champion["extra"]["kind"]
    rp = params["retrain"]
    hold = snap[(snap["as_of_date"] >= pd.Timestamp(rp["holdout_start"])) & (snap["as_of_date"] <= pd.Timestamp(rp["holdout_end"]))]
    calm = snap[snap["split"] == "test"]                                 # the calm 2022-12..2023-02 months ("the economy recovers")
    out = {}
    for name, tr in (("rolling", train_rolling), ("stress_only", train_stress)):
        tr = tr[~tr.index.isin(calm.index)]                    # keep the calm months OUT of training so the calm check is out-of-sample for both
        pipe = fit_model(tr, feats, kind, params)
        out[name] = dict(n_train=len(tr), positives=int(tr[TARGET].sum()),
                         stress_holdout=evaluate_pipeline(pipe, hold, feats, TARGET, top)[0],
                         calm_period=evaluate_pipeline(pipe, calm, feats, TARGET, top)[0])
    print(f"\n   {'window':<13}{'train rows':>11}{'defaulters':>11}{'AUC stress hold-out':>21}{'AUC calm period':>17}")
    for name, r in out.items():
        print(f"   {name:<13}{r['n_train']:>11,}{r['positives']:>11,}{r['stress_holdout']['roc_auc']:>21.4f}{r['calm_period']['roc_auc']:>17.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Drift check -> decision -> challenger -> compare -> approval -> promotion")
    ap.add_argument("--prod", default="data/production/batch_drifted.csv", help="the production batch that triggered the review")
    ap.add_argument("--today", default=None, help="simulated current date (default: params retrain.today)")
    ap.add_argument("--window", choices=["rolling", "stress_only"], default="rolling")
    ap.add_argument("--version", default=None, help="challenger version (default: next number after the champion)")
    ap.add_argument("--approve-as", default=None, metavar="NAME", help="simulate the human sign-off; without it we stop at 'awaiting approval'")
    ap.add_argument("--scheduled", action="store_true", help="the scheduled quarterly retrain is due")
    ap.add_argument("--train-anyway", action="store_true", help="CLASSROOM ONLY: continue even if the ladder did not say RETRAIN_CANDIDATE")
    ap.add_argument("--calm-check", action="store_true", help="also compare windows on calm data (failure scenario c)")
    ap.add_argument("--min-improvement", type=float, default=None, help="override the required ROC-AUC gain (demo: a high bar makes the rules fail)")
    ap.add_argument("--rollback", action="store_true", help="restore the previous champion and exit")
    a = ap.parse_args()

    if a.rollback:
        before = load_artifact(CHAMPION_PATH)["version"]
        prev = promotion.rollback()
        print(f"ROLLBACK: champion was {before}; restored previous version -> {prev}" if prev else "ROLLBACK: nothing archived, nothing to restore")
        if prev:
            print(json.dumps(json.loads(REGISTRY_PATH.read_text()), indent=2))
        return

    params = load_params()
    rp, top = params["retrain"], params["model"]["top_pct"]
    today = a.today or rp["today"]
    rp = {**rp, "today": today}
    if a.min_improvement is not None:
        rp["min_improvement"] = a.min_improvement
    champion = load_artifact(CHAMPION_PATH)

    print("=" * 100)
    print(f"RETRAINING REVIEW   champion = {champion['version']} ({champion['extra']['kind']})   today = {today}   batch = {a.prod}")
    print("=" * 100)
    print("\nSTEPS 1-3  data quality -> drift -> decision ladder")
    res = run_drift_check(a.prod, str(DATA_PROCESSED / "reference.csv"), today, a.scheduled, params["drift"]["alpha"], params["drift"]["min_effect"], quiet=True)
    print(f"   data quality : {'OK' if res['data_quality']['ok'] else 'FAILED ' + str(res['data_quality']['problems'])}")
    d = pd.DataFrame(res["drift"])
    print(f"   KS test (H0: same distribution, alpha={params['drift']['alpha']}, large = D >= {params['drift']['min_effect']}):")
    for _, r in d.iterrows():
        print(f"      {r['feature']:<26} D={r['ks_statistic']:.3f}  p={r['p_value']:.1e}  {r['status']}")
    dec = res["decision"]
    print(f"   DECISION     : {dec['decision']}")
    for r in dec["reasons"]:
        print("      *", r)
    if dec["decision"] != "RETRAIN_CANDIDATE" and not a.train_anyway:
        print(f"\nSTOP: the ladder says {dec['decision']}, not RETRAIN_CANDIDATE.  No challenger is trained.  (Drift alone never triggers retraining.)")
        return
    if dec["decision"] != "RETRAIN_CANDIDATE":
        print("\n   !! --train-anyway: CLASSROOM ONLY - a real scheduled job stops here.")

    snap = load_snapshots()
    train = training_window(snap, rp, a.window)
    hold = mature(snap[(snap["as_of_date"] >= pd.Timestamp(rp["holdout_start"])) & (snap["as_of_date"] <= pd.Timestamp(rp["holdout_end"]))], today)
    version = a.version or next_version(champion["version"])
    print(f"\nSTEP 4  train challenger {version}   window={a.window}")
    print(f"   training rows {len(train):,}  as_of {train['as_of_date'].min().date()} .. {train['as_of_date'].max().date()}  "
          f"defaulters {int(train[TARGET].sum())} ({train[TARGET].mean():.2%})")
    print(f"   hold-out rows {len(hold):,}  as_of {hold['as_of_date'].min().date()} .. {hold['as_of_date'].max().date()}  "
          f"(newer than every training row; purge gap = {_months_between(rp['train_end'], rp['holdout_start'])} months)")
    ch = train_challenger(train, snap, champion, version, rp, hold, top, params)

    print("\nSTEP 5  champion vs challenger on the SAME newer hold-out (labels known; neither model trained on these rows)")
    champ_hold, champ_scores = evaluate_pipeline(champion["pipeline"], hold, champion["input_columns"], TARGET, top)
    _print_compare(f"champion {champion['version']}", champ_hold, f"challenger {version}", ch["metrics"])
    print("   what the collections team sees (champion uses ITS OWN stored contact threshold, challenger its own):")
    for name, scores, thr in ((f"champion {champion['version']}", champ_scores, champion["threshold"]), (f"challenger {version}", ch["scores"], ch["threshold"])):
        b = business_summary(hold[TARGET], scores, top)
        print(f"      {name:<16} top-10% list reaches {b['defaulters_reached']}/{b['true_defaulters']} defaulters ({b['share_of_defaulters_reached']:.0%}); "
              f"score >= its threshold flags {float((scores >= thr).mean()):.1%} of customers (plan {top:.0%})")

    cmp_ = promotion.compare(champ_hold, ch["metrics"], primary="roc_auc", min_improvement=rp["min_improvement"], guardrails=rp["guardrails"])
    print(f"\nSTEP 6  promotion rules (promotion.compare):  ROC-AUC delta {cmp_['delta']:+.4f}")
    for k, v in cmp_["checks"].items():
        print(f"      [{'PASS' if v else 'FAIL'}] {k}")
    print(f"   rules say: {'PROMOTE' if cmp_['promote'] else 'DO NOT PROMOTE'}")

    record = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), champion=champion["version"], challenger=version,
                  window=a.window, today=today, decision_ladder=dec["decision"], holdout=f"{hold['as_of_date'].min().date()}..{hold['as_of_date'].max().date()}",
                  champion_holdout=champ_hold, challenger_holdout=ch["metrics"], checks=cmp_["checks"], rules_say_promote=cmp_["promote"],
                  approver=a.approve_as, outcome=None)
    save_artifact(CHALLENGER_PATH, ch["pipe"], version, ch["feats"], ch["metrics"], threshold=ch["threshold"],
                  extra=dict(kind=ch["kind"], features=ch["feats"], window=a.window, train_rows=len(train), holdout=record["holdout"],
                             val_metrics=champion["extra"].get("val_metrics"), test_base_rate=ch["metrics"]["base_rate"],
                             test_mean_pd=ch["metrics"]["mean_score"], train_period=f"{train['as_of_date'].min().date()}..{train['as_of_date'].max().date()}"))
    print(f"\nSTEP 7  APPROVAL RECORD  (saved to reports/approval_record.json)")
    print(f"   proposed by : retraining job (decision ladder = {dec['decision']})")
    print(f"   change      : champion {champion['version']} -> challenger {version}   (window {a.window}, {len(train):,} training rows)")
    print(f"   evidence    : hold-out ROC-AUC {champ_hold['roc_auc']} -> {ch['metrics']['roc_auc']}, recall@10% {champ_hold['recall_at_10pct']} -> {ch['metrics']['recall_at_10pct']}")
    print(f"   rules       : {'ALL PASSED' if cmp_['promote'] else 'FAILED: ' + ', '.join(k for k, v in cmp_['checks'].items() if not v)}")
    print(f"   approver    : {a.approve_as or '<< none yet >>'}")

    print("\nSTEP 8  promotion")
    if not cmp_["promote"]:
        record["outcome"] = "REJECTED_BY_RULES"
        print("   NOT PROMOTED: the rules failed.  The champion keeps serving.  (A human signature cannot override a failed guardrail here.)")
    elif not a.approve_as:
        record["outcome"] = "AWAITING_APPROVAL"
        print("   AWAITING APPROVAL: rules passed, but promotion needs a human sign-off.  Re-run with  --approve-as \"Name (role)\".")
    else:
        reason = (f"hold-out ROC-AUC {champ_hold['roc_auc']} -> {ch['metrics']['roc_auc']}; window={a.window}; "
                  f"decision ladder {dec['decision']}")
        new = promotion.promote_challenger(a.approve_as, reason)
        record["outcome"] = f"PROMOTED_{new}"
        print(f"   PROMOTED: {new} is now the champion (approved by {a.approve_as}).  The previous champion is archived: models/model_{champion['version']}.joblib")
        print("   models/registry.json:")
        for h in json.loads(REGISTRY_PATH.read_text()):
            print(f"      {h['version']:<4} {h['stage']:<10} approved_by={h['approved_by']!r:<28} roc_auc={h['metrics'].get('roc_auc')}")
        print("   (a registry roc_auc is measured on THAT model's own evaluation period: v1 = calm test months, v2 = the stress hold-out.  "
              "Compare models only on the SAME hold-out, as in step 5.)")
        ref_out = DATA_PROCESSED / f"reference_{new}.csv"
        train_ref = train.copy()
        from src.features.definitions import MODEL_FEATURES
        train_ref[MODEL_FEATURES].sample(min(params["reference_sample_rows"], len(train_ref)), random_state=0).to_csv(ref_out, index=False)
        print(f"   NEXT: the new model's drift reference = {ref_out.name} (its training data).  Copy it over data/processed/reference.csv when you deploy {new}.")
        print("   To undo:  python -m src.models.retrain --rollback")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "approval_record.json").write_text(json.dumps(record, indent=2, default=str))
    save_fact(f"retrain_{a.window}_{record['outcome']}", dict(record, drift_table=res["drift"], decision_reasons=dec["reasons"]))

    if a.calm_check:
        print("\nFAILURE SCENARIO (c): is a stress-only training window a good idea?  Both windows, same model family, same hold-outs")
        print("   (the calm months 2022-12..2023-02 are removed from BOTH training sets so that the calm column is out-of-sample)")
        r = calm_check(champion, training_window(snap, rp, "rolling"), training_window(snap, rp, "stress_only"), snap, params, top)
        d_calm = r["stress_only"]["calm_period"]["roc_auc"] - r["rolling"]["calm_period"]["roc_auc"]
        d_str = r["stress_only"]["stress_holdout"]["roc_auc"] - r["rolling"]["stress_holdout"]["roc_auc"]
        print(f"   Reading: stress_only minus rolling = {d_str:+.4f} AUC on the stress hold-out and {d_calm:+.4f} on calm data.  It learned from "
              f"{r['stress_only']['n_train'] / r['rolling']['n_train']:.0%} of the rows and ONE regime: no gain where we hoped, a loss where the economy normalises.")
        print("   Lesson: choose the window to cover every regime the model will meet; 'train on the latest months only' is a hypothesis to TEST, not a default.")


if __name__ == "__main__":
    main()
