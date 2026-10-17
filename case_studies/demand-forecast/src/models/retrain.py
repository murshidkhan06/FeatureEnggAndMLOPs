"""RETRAINING WORKFLOW:  production batch -> data quality -> drift (KS) -> decision ladder -> challenger -> same hold-out
-> promotion rules -> approval record -> (human sign-off) -> promote -> rollback demo.

    python -m src.models.retrain --prod data/production/batch_drifted.csv                       # stops at "awaiting approval"
    python -m src.models.retrain --prod data/production/batch_drifted.csv --approve-as Asha     # human sign-off, then promote
    python -m src.models.retrain --prod data/production/batch_drifted.csv --approve-as Asha --demo-rollback
    python -m src.models.retrain --prod data/production/batch_drifted.csv --compare-strategies  # rolling vs expanding window
    python -m src.models.retrain --prod data/production/batch_normal.csv  --mode scheduled      # the monthly job (no trigger needed)
    python -m src.models.retrain --prod data/production/batch_drifted.csv --force-model hgb --approve-as Asha   # guardrail demo

THE RETRAINING STRATEGY (configs/retrain_policy.yaml, coded below)
  WEEKLY      refresh the history store with last week's actual sales + run the batch forecast (NO new model: only fresh inputs).
  MONTHLY     scheduled retrain on a ROLLING window (last 12 months) or an EXPANDING window (everything so far).   --mode scheduled
  TRIGGER     weekly WAPE >= 25% worse than validation WAPE in 2 weeks in a row AND the lag/rolling features really drifted.
              --mode trigger (default): the decision ladder must say RETRAIN_CANDIDATE, otherwise we stop here.

DRIFT IS A SIGNAL, NOT A COMMAND.  A retrain only starts when evidence says the model fails (trigger) or when the calendar says
it is the monthly slot (scheduled).  Either way the CHALLENGER must still win a fair comparison and a human must sign off.

THE FAIR COMPARISON (the part that is easy to get wrong)
        |---------- history before the validation window ----------|-- VALID 14 d --|-- HOLD-OUT 28 d --|
        challenger: fit here (rolling: last 12 months only)          pick Ridge/HGB     champion AND challenger are scored on
        (then re-fit on history + VALID)                                                the SAME most recent 4 weeks
  The champion is NOT re-fitted: it is the model that is serving now.  Both are judged on days neither of them has seen.
  Promotion rule: challenger WAPE at least 0.5 points lower AND |bias| <= 8% AND bias not > 1 point worse than the champion.
  Worked example: champion WAPE 0.21 (bias -17%), challenger 0.12 (bias -6%)  ->  gain 9 points, |bias| 6% <= 8%  ->  eligible.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common import promotion
from src.common.artifact import load_artifact, save_artifact
from src.common.facts import save_fact
from src.common.params import load_params, load_policy
from src.common.paths import CHALLENGER_PATH, CHAMPION_PATH, DATA_RAW, REGISTRY_PATH, REPORTS
from src.data.make_dataset import from_batch
from src.features import data_quality
from src.features.build_features import training_frame
from src.features.definitions import HISTORY_COLUMNS
from src.features.registry import check_features_approved
from src.models import metrics as M
from src.models.pipelines import MODEL_NAMES, build_pipeline
from src.models.train import fit_and_select, interval_quantiles, score
from src.monitoring.run_drift_check import assess


# --------------------------------------------------------------------------------------------- data windows
def refreshed_frame(batch_path: str) -> pd.DataFrame:
    """Training history + the labelled production batch -> ONE engineered frame (lags are computed on the joined timeline,
    so the first production days get their lags from the last training days: exactly what the nightly refresh does)."""
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    batch = from_batch(pd.read_csv(batch_path))
    both = pd.concat([raw[HISTORY_COLUMNS], batch[HISTORY_COLUMNS]], ignore_index=True)
    return training_frame(both)


def make_windows(df: pd.DataFrame, strategy: str, holdout_days: int, valid_days: int, window_months: int) -> dict:
    """Chronological windows for the challenger.   strategy = 'rolling' (last N months) or 'expanding' (everything).

    Example: data ends 2025-02-28, holdout 28 d, valid 14 d:
        hold-out 2025-02-01..02-28 | valid 2025-01-18..01-31 | train: rolling 2024-01-18..2025-01-17, expanding 2023-01-31..2025-01-17"""
    end = df["date"].max()
    ho_start = end - pd.Timedelta(days=holdout_days - 1)
    va_start = ho_start - pd.Timedelta(days=valid_days)
    hold, valid, past = df[df["date"] >= ho_start], df[(df["date"] >= va_start) & (df["date"] < ho_start)], df[df["date"] < va_start]
    if strategy == "rolling":
        past = past[past["date"] >= va_start - pd.DateOffset(months=window_months)]
    elif strategy != "expanding":
        raise ValueError("strategy must be 'rolling' or 'expanding'")
    assert past["date"].max() < valid["date"].min() <= valid["date"].max() < hold["date"].min(), "windows must be chronological"
    return dict(train=past, valid=valid, hold=hold)


def holdout_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    """The numbers the promotion rule compares (abs_bias_pct = |bias| as a share of units sold; lower is better for both)."""
    m = M.report(frame["sales"], pred, frame["rolling_mean_30"])
    m["abs_bias_pct"] = round(abs(m["bias_pct"]), 4)
    return m


# --------------------------------------------------------------------------------------------- challenger
def build_challenger(df: pd.DataFrame, features: list[str], params: dict, policy: dict, strategy: str,
                     force_model: str | None = None, verbose: bool = True) -> dict:
    """Fit the challenger on a window and score it on the hold-out.  Returns pipeline + metrics + the windows used."""
    check_features_approved(features)                                    # FEATURE GATE again: a retrain may not smuggle in a Rejected feature
    w = make_windows(df, strategy, policy["holdout_days"], policy["valid_days"], policy["monthly_scheduled"]["window_months"])
    sel = fit_and_select(w["train"], w["valid"], features, params, force_model, verbose=verbose)
    best = sel["best"]
    both = pd.concat([w["train"], w["valid"]])
    pipe = build_pipeline(best, features, params).fit(both[features], both["sales"])
    ho_pred = score(pipe, w["hold"], features)
    q = interval_quantiles(w["valid"]["sales"].values, sel["results"][best]["valid_pred"], params["interval"]["low_quantile"],
                           params["interval"]["high_quantile"])
    m = holdout_metrics(w["hold"], ho_pred)
    m.update(val_wape=round(sel["results"][best]["wape"], 4), val_bias_pct=round(sel["results"][best]["bias_pct"], 4))
    return dict(strategy=strategy, model=best, pipe=pipe, metrics=m, interval=q, windows=w, valid_results=
                {k: {"wape": r["wape"], "bias_pct": r["bias_pct"]} for k, r in sel["results"].items()})


def span(d: pd.DataFrame) -> str:
    return f"{d['date'].min().date()}..{d['date'].max().date()} ({len(d):,} rows)"


def compare_strategies(df, features, params, policy, force_model=None) -> dict:
    """Same data, same hold-out: rolling vs expanding.  Informational: the strategy used for the real job comes from the policy."""
    rows = {}
    for s in ("rolling", "expanding"):
        c = build_challenger(df, features, params, policy, s, force_model, verbose=False)
        rows[s] = dict(model=c["model"], train_rows=len(c["windows"]["train"]), valid_wape=c["metrics"]["val_wape"],
                       holdout_wape=c["metrics"]["wape"], holdout_bias_pct=c["metrics"]["bias_pct"])
    return rows


# --------------------------------------------------------------------------------------------- approval record
def decide_promotion(champ: dict, chall: dict, policy: dict) -> dict:
    p = policy["promotion"]
    g = p["guardrails"]
    res = promotion.compare(champ, chall, primary=p["primary_metric"], min_improvement=p["min_improvement"],
                            guardrails={"abs_bias_pct": g["max_bias_worsening"]}, lower_is_better=("wape", "abs_bias_pct"),
                            absolute_limits={"abs_bias_pct": g["max_abs_bias_pct"]})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prod", required=True, help="labelled production batch (CSV with the true sales)")
    ap.add_argument("--mode", choices=["trigger", "scheduled"], default="trigger")
    ap.add_argument("--strategy", choices=["rolling", "expanding"], default=None, help="default: from retrain_policy.yaml")
    ap.add_argument("--compare-strategies", action="store_true")
    ap.add_argument("--force-model", choices=MODEL_NAMES, default=None, help="skip the Ridge-vs-HGB choice (used in the guardrail demo)")
    ap.add_argument("--approve-as", default=None, metavar="NAME", help="simulates the HUMAN sign-off; without it we stop at 'awaiting approval'")
    ap.add_argument("--demo-rollback", action="store_true", help="after promotion, roll back to the previous champion")
    ap.add_argument("--version", default=None)
    a = ap.parse_args()

    params, policy = load_params(), load_policy()
    strategy = a.strategy or policy["monthly_scheduled"]["strategy"]
    champ_art = load_artifact(CHAMPION_PATH)
    features = champ_art["input_columns"]
    print(f"RETRAIN JOB  mode={a.mode}  strategy={strategy}  batch={a.prod}")
    print(f"serving champion: {champ_art['version']} ({champ_art['extra']['model_name']}), trained through {champ_art['extra']['trained_through']}\n")

    # ---- step 1+2+3+4: data quality, drift, labelled performance, business impact (the ladder from run_drift_check) -------
    print("=" * 100 + "\nSTEP 1-4  EVIDENCE (data quality -> drift -> labelled performance -> business impact)\n" + "=" * 100)
    ev = assess(a.prod, verbose=False)
    dq = ev.get("data_quality", {})
    if ev["decision"]["decision"] == "FIX_DATA_PIPELINE":
        print("DATA QUALITY FAILED:")
        for p in ev["decision"]["reasons"]:
            print("  *", p)
        print("\nDECISION: FIX_DATA_PIPELINE - no retraining: a model trained on broken data would be worse, not better.")
        return
    print("data quality: OK")
    t = pd.DataFrame(ev["table"])[["feature", "ks_statistic", "p_value", "status"]]
    print("\nKS drift (H0: same distribution, alpha=0.05, practical threshold D>=0.10):")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    wk = pd.DataFrame(ev["weekly"])[["week", "from_date", "wape", "bias_pct", "vs_validation", "breach"]]
    print("\nweekly labelled performance of the CHAMPION:")
    print(wk.to_string(index=False))
    print(f"\nlag/rolling drift: {'yes' if ev['lag_drift'] else 'no'};  worst-of-last-2-weeks WAPE worsening: {ev['perf_drop']:+.1%} "
          f"(trigger >= {policy['trigger']['wape_worsen_pct']:.0%});  business impact: {'HIGH' if ev['business']['impact_high'] else 'LOW'}")
    print(f"LADDER DECISION: {ev['decision']['decision']}")
    for r in ev["decision"]["reasons"]:
        print("  *", r)
    if a.mode == "trigger" and ev["decision"]["decision"] != "RETRAIN_CANDIDATE":
        print(f"\nSTOP: trigger mode retrains only on RETRAIN_CANDIDATE (got {ev['decision']['decision']}).  "
              "Drift alone is not a reason; the monthly scheduled job (--mode scheduled) will still run on its own calendar.")
        save_fact("retrain_" + a.mode, dict(prod=a.prod, stopped_at="ladder", ladder=ev["decision"]))
        return
    if a.mode == "scheduled":
        print("\nSCHEDULED mode: the monthly job retrains regardless of the ladder outcome (data quality passed).")

    # ---- step 5: challenger on refreshed data ------------------------------------------------------------------------
    print("\n" + "=" * 100 + f"\nSTEP 5  TRAIN CHALLENGER on refreshed data ({strategy} window)\n" + "=" * 100)
    df = refreshed_frame(a.prod)
    cs = None
    if a.compare_strategies:
        cs = compare_strategies(df, features, params, policy, a.force_model)
        print("ROLLING (last 12 months) vs EXPANDING (all history) on the SAME hold-out:")
        print(pd.DataFrame(cs).T.to_string())
        print("(informational; the strategy for this job is the one in retrain_policy.yaml / --strategy)\n")
    ch = build_challenger(df, features, params, policy, strategy, a.force_model)
    w = ch["windows"]
    print(f"train   {span(w['train'])}\nvalid   {span(w['valid'])}   -> chose '{ch['model']}'\nhold-out {span(w['hold'])}   (most recent {policy['holdout_days']} days: never used for fitting)")

    # ---- step 6: champion AND challenger on the SAME hold-out --------------------------------------------------------
    print("\n" + "=" * 100 + "\nSTEP 6  CHAMPION vs CHALLENGER on the same most-recent-4-weeks hold-out\n" + "=" * 100)
    champ_pred = score(champ_art["pipeline"], w["hold"], features)
    cm = holdout_metrics(w["hold"], champ_pred)
    chm = ch["metrics"]
    tab = pd.DataFrame({"champion": cm, "challenger": {k: chm[k] for k in cm}}).T[["wape", "bias_pct", "abs_bias_pct", "mae", "rmse"]]
    print(tab.to_string())
    units = int(w["hold"]["sales"].sum())
    print(f"\n{units:,} units were sold in the hold-out.  Champion bias {cm['bias_pct']:+.1%} = {abs(cm['bias']) * len(w['hold']):,.0f} units "
          f"{'short' if cm['bias'] < 0 else 'surplus'}; challenger bias {chm['bias_pct']:+.1%}.")

    # ---- step 7: promotion rules -------------------------------------------------------------------------------------
    print("\n" + "=" * 100 + "\nSTEP 7  PROMOTION RULES (with guardrails)\n" + "=" * 100)
    res = decide_promotion(cm, chm, policy)
    for k, ok in res["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"  WAPE improvement over champion: {res['delta']:+.4f}")

    version = a.version or (f"v{len(json.loads(REGISTRY_PATH.read_text())) + 1}-{strategy}" if REGISTRY_PATH.exists() else f"v2-{strategy}")
    extra = dict(model_name=ch["model"], features=features, interval=ch["interval"], train_rows=len(w["train"]) + len(w["valid"]),
                 trained_through=str(w["valid"]["date"].max().date()), strategy=strategy, mode=a.mode,
                 holdout=dict(start=str(w["hold"]["date"].min().date()), end=str(w["hold"]["date"].max().date())),
                 candidates_valid=ch["valid_results"])
    save_artifact(CHALLENGER_PATH, ch["pipe"], version, features, chm, threshold=0.0, extra=extra)

    # ---- step 8: approval record ---------------------------------------------------------------------------------------
    record = dict(record_id=f"retrain-{datetime.now(timezone.utc):%Y%m%d}-{version}", champion=champ_art["version"], challenger=version,
                  challenger_model=ch["model"], strategy=strategy, mode=a.mode, trigger=ev["decision"]["decision"],
                  holdout=extra["holdout"], champion_metrics={k: cm[k] for k in ("wape", "bias_pct")},
                  challenger_metrics={k: chm[k] for k in ("wape", "bias_pct")}, rules_pass=res["promote"], checks=res["checks"],
                  approved_by=a.approve_as if res["promote"] else None, signoff_offered_by=a.approve_as, features_gate="all features Approved in configs/feature_registry.yaml")
    print("\n" + "=" * 100 + "\nSTEP 8  APPROVAL RECORD\n" + "=" * 100)
    print(json.dumps(record, indent=2))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "approval_record.json").write_text(json.dumps(record, indent=2))
    outcome = "NOT_PROMOTED"
    if not res["promote"]:
        failed = [k for k, ok in res["checks"].items() if not ok]
        print(f"\nOUTCOME: NOT PROMOTED.  Failed: {failed}.  A human sign-off cannot override a failed rule; the champion keeps serving.")
    elif not a.approve_as:
        outcome = "AWAITING_APPROVAL"
        print("\nOUTCOME: AWAITING APPROVAL.  The rules pass, but a person must sign off.\n"
              f"  -> python -m src.models.retrain --prod {a.prod} --mode {a.mode} --approve-as <your name>")
    else:
        reason = (f"{a.mode}/{strategy}: hold-out WAPE {cm['wape']:.3f} -> {chm['wape']:.3f}, bias {cm['bias_pct']:+.1%} -> {chm['bias_pct']:+.1%}")
        new = promotion.promote_challenger(a.approve_as, reason)
        outcome = "PROMOTED"
        print(f"\nOUTCOME: PROMOTED {new} to champion (approved by {a.approve_as}); the old champion is archived as models/model_{champ_art['version']}.joblib")
        print("  restart / reload the API (or roll the Kubernetes deployment) so that it serves the new champion.")
        if a.demo_rollback:
            print("\nROLLBACK DEMO: suppose monitoring shows the new model behaves badly in production ...")
            prev = promotion.rollback()
            print(f"  promotion.rollback() -> champion is {load_artifact(CHAMPION_PATH)['version']} again (restored: {prev})")
            outcome = "PROMOTED_THEN_ROLLED_BACK"
        print("\nREGISTRY (models/registry.json):")
        for h in json.loads(REGISTRY_PATH.read_text()):
            print(f"  {h['version']:<18} {h['stage']:<10} approved_by={h['approved_by']:<10} {h['reason'][:80]}")
    save_fact("retrain_" + a.mode + "_" + strategy + ("_" + a.force_model if a.force_model else "") + "_" + outcome.lower(),
              dict(prod=a.prod, outcome=outcome, record=record, champion_holdout=cm, challenger_holdout=chm,
                   windows={k: span(v) for k, v in w.items()}, delta=res["delta"], checks=res["checks"],
                   strategies=cs))


if __name__ == "__main__":
    main()
