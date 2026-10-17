"""Batch drift check + retraining decision (what a scheduled job would run every week).

    python -m src.monitoring.run_drift_check --prod data/production/batch_drifted.csv
    python -m src.monitoring.run_drift_check --prod data/production/batch_festival.csv --ref-window seasonal

THE KS TEST IN ONE PARAGRAPH
  Reference = what the model saw in training.  Production = what it sees now.
  H0 (null hypothesis): both samples come from the same distribution.  D = the largest vertical gap between the two empirical
  CDFs (the EFFECT SIZE, 0..1); p-value = how surprising a gap this big would be if H0 were true.  alpha = 0.05.
  p < alpha  -> reject H0 (statistically significant).  With thousands of rows even tiny gaps are significant, so we ALSO require
  D >= 0.10 (practically significant).  "Significant but small" = real but not worth acting on.

IS A DISTRIBUTION SHIFT AUTOMATICALLY A MODEL FAILURE?   NO.
  The model's INPUTS can move and the model can still be right, because lag features move with the level
  (festival batch: inputs drift, accuracy holds).  And the opposite: a shift that the model cannot follow does hurt
  (drifted batch: tree under-forecasts).  So the ladder looks at FOUR kinds of evidence, cheapest first:
     1 data quality   2 drift (KS)   3 labelled performance (weekly WAPE, bias)   4 business impact (units / stock cost)
  Drift alone never triggers retraining.

SEASONAL REFERENCE WINDOW (--ref-window seasonal)
  Comparing Diwali week with an average training week is unfair: demand SHOULD rise.  The seasonal reference keeps only the
  training rows from the same time of year (+-14 days) in earlier years, so only the UNEXPECTED part of the shift remains.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.drift import explain_row, ks_drift_report
from src.common.facts import save_fact
from src.common.params import load_policy
from src.common.paths import CHAMPION_PATH, DATA_RAW, REFERENCE_PATH, REPORTS
from src.common.retrain_decision import apply_forecast_policy, decide
from src.features import data_quality
from src.features.definitions import LAGS
from src.models.train import score
from src.monitoring.performance import labelled_features, trigger_status, weekly_performance


def seasonal_reference(ref: pd.DataFrame, prod_dates: pd.Series, window_days: int = 14) -> pd.DataFrame:
    """Training rows whose day-of-year is within +-window_days of ANY day of the production batch (circular year)."""
    ref = ref.assign(date=pd.to_datetime(ref["date"]))
    doy = ref["date"].dt.dayofyear.values[:, None]
    pdoy = pd.to_datetime(prod_dates).dt.dayofyear.drop_duplicates().values[None, :]
    diff = np.abs(doy - pdoy)
    diff = np.minimum(diff, 365 - diff)
    return ref[(diff <= window_days).any(axis=1)]


def assess(prod: str, ref_path: str = str(REFERENCE_PATH), ref_window: str = "all", alpha: float = 0.05, min_effect: float = 0.10,
           perf_drop_override: float | None = None, impact_override: str | None = None, verbose: bool = True) -> dict:
    """Run the 4-step ladder on one production batch.  Returns the payload that is also written to reports/drift_report.json."""
    say = print if verbose else (lambda *a, **k: None)
    policy = load_policy()
    pol = policy["trigger"]
    batch = pd.read_csv(prod)
    say(f"PRODUCTION BATCH {prod}: {len(batch):,} rows, {batch.forecast_date.min()} .. {batch.forecast_date.max()}\n")

    say("STEP 1 - DATA QUALITY")
    dq = data_quality.validate(batch)
    if verbose:
        data_quality.print_report(dq)

    ref = pd.read_csv(ref_path)
    if ref_window == "seasonal":
        before = len(ref)
        ref = seasonal_reference(ref, batch["forecast_date"])
        say(f"\nREFERENCE: seasonal window (same time of year, +-14 days): {len(ref):,} of {before:,} training rows")
    else:
        say(f"\nREFERENCE: all training rows ({len(ref):,} sampled rows, level 100 -> 130)")
    if not dq["ok"]:
        say("\nDECISION: FIX_DATA_PIPELINE  (retraining on broken data would make things worse)")
        for p in dq["problems"]:
            say("  *", p)
        return dict(prod=prod, decision=dict(decision="FIX_DATA_PIPELINE", reasons=dq["problems"]), data_quality=dict(ok=False, problems=dq["problems"]))

    f = labelled_features(prod)
    feats = [c for c in P.DRIFT_FEATURES if c in f.columns and c in ref.columns] + (["sales"] if "sales" in ref.columns else [])
    rep = ks_drift_report(ref, f, feats, alpha=alpha, min_effect=min_effect)
    say("\nSTEP 2 - DRIFT (two-sample KS test)")
    say("H0: reference and production values come from the same distribution;   alpha =", alpha,
        "  practical threshold D >=", min_effect, "\n")
    used = set(load_artifact(CHAMPION_PATH)["input_columns"])
    show = rep.assign(used_by_champion=rep.feature.isin(used)).rename(columns={"statistically_significant": "signif", "practically_significant": "large"})
    say(show.to_string(index=False))
    say("(drift in a column the champion does not use - e.g. absolute price, or the target itself - is information, not a threat)")
    say()
    for _, r in rep.iterrows():
        say(" -", explain_row(r, alpha))
    lag_drift = bool(rep[rep.feature.isin(LAGS)]["status"].eq("drift").any())

    say("\nSTEP 3 - LABELLED PERFORMANCE (labels arrive the next day in retail)")
    art = load_artifact(CHAMPION_PATH)
    pred = score(art["pipeline"], f, art["input_columns"])
    wk = weekly_performance(f, pred, art["metrics"]["val_wape"], pol["wape_worsen_pct"])
    st = trigger_status(wk, pol["consecutive_weeks"])
    say(wk[["week", "from_date", "wape", "bias_pct", "vs_validation", "breach"]].to_string(index=False))
    perf_drop = perf_drop_override if perf_drop_override is not None else st["perf_drop"]
    say(f"validation WAPE {art['metrics']['val_wape']:.3f}; smallest worsening over the last {pol['consecutive_weeks']} weeks = {perf_drop:+.1%} "
        f"(trigger: >= {pol['wape_worsen_pct']:.0%})")

    say("\nSTEP 4 - BUSINESS IMPACT")
    under = max(0.0, float((f.sales - pred).clip(lower=0).sum()))
    over = max(0.0, float((pred - f.sales).clip(lower=0).sum()))
    impact_high = abs(st["last_weeks_bias_pct"]) >= policy["business_impact_bias_pct"]
    if impact_override:
        impact_high = impact_override == "yes"
    say(f"under-forecast (lost-sales risk) {under:,.0f} units; over-forecast (waste risk) {over:,.0f} units; "
        f"bias of the last weeks {st['last_weeks_bias_pct']:+.1%} -> impact {'HIGH' if impact_high else 'LOW'} "
        f"(threshold |bias| >= {policy['business_impact_bias_pct']:.0%})")

    out = decide(rep[rep.feature != "sales"], data_quality_ok=True, perf_drop=perf_drop, perf_drop_threshold=pol["wape_worsen_pct"],
                 labels_available=True, business_impact_high=impact_high)
    out = apply_forecast_policy(out, lag_drift, pol["require_lag_drift"])
    say("\nDECISION LADDER RESULT")
    say("DATA QUALITY: OK")
    say("DECISION:", out["decision"])
    for r in out["reasons"]:
        say("  *", r)
    n_drift = int((rep.status == "drift").sum())
    say(f"\nSUMMARY: {n_drift} of {len(rep)} tested columns show significant+large drift (lag drift: {'yes' if lag_drift else 'no'}); "
        f"performance evidence: {'yes' if perf_drop >= pol['wape_worsen_pct'] else 'no'}.")
    if n_drift and out["decision"] in ("MONITOR", "NO_ACTION"):
        say("-> Distribution shift WITHOUT model failure: the lags followed the level.  Not a reason to retrain.")
    if out["decision"] == "RETRAIN_CANDIDATE":
        say("-> Shift AND failure: train a CHALLENGER (python -m src.models.retrain ...).  It must still beat the champion to be promoted.")
    payload = dict(prod=prod, ref_window=ref_window, table=rep.to_dict("records"), decision=out, weekly=wk.to_dict("records"),
                   data_quality=dict(ok=dq["ok"], checks=[list(c) for c in dq["checks"]]), perf_drop=perf_drop, lag_drift=lag_drift,
                   business=dict(under_units=under, over_units=over, impact_high=impact_high))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--ref", default=str(REFERENCE_PATH))
    ap.add_argument("--ref-window", choices=["all", "seasonal"], default="all")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-effect", type=float, default=0.10)
    ap.add_argument("--perf-drop", type=float, default=None, help="override: observed relative WAPE worsening (default: measured)")
    ap.add_argument("--business-impact-high", choices=["yes", "no"], default=None)
    a = ap.parse_args()
    payload = assess(a.prod, a.ref, a.ref_window, a.alpha, a.min_effect, a.perf_drop, a.business_impact_high)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift_report.json").write_text(json.dumps(payload, indent=2, default=str))
    name = a.prod.replace("\\", "/").split("/")[-1].replace(".csv", "") + ("_seasonal" if a.ref_window == "seasonal" else "")
    save_fact("drift_" + name, payload)


if __name__ == "__main__":
    main()
