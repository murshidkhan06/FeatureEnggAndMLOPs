"""Batch drift check + retraining DECISION LADDER (what a scheduled monitoring job would run).

python -m src.monitoring.run_drift_check --prod data/production/batch_drifted.csv [--labels-available] [--perf-drop 0.05]

Drift is a SIGNAL, not a command.  The script gathers FOUR kinds of evidence and then applies the ladder
(src/common/retrain_decision.py):
   1. DATA QUALITY      is the batch healthy?  (src/monitoring/data_quality.py)      -> if not: FIX_DATA_PIPELINE
   2. DRIFT (KS test)   H0: reference and production values come from the SAME distribution.
                        D = largest gap between the two empirical CDFs; p-value = how surprising such a gap would be if H0 were true.
                        alpha = 0.05.  Large samples make tiny gaps "significant", so we ALSO require D >= 0.10 to call it practical drift.
   3. PERFORMANCE       only if labels have arrived (--labels-available): AUC on real outcomes vs the model's test AUC
   4. BUSINESS IMPACT   share of applicants per decision band and (if labelled) predicted vs observed default rate
Exit message tells you what to do next; nothing is retrained here.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.drift import explain_row, ks_drift_report
from src.common.paths import CHAMPION_PATH, REFERENCE_PATH, REPORTS
from src.common.retrain_decision import decide
from src.features.build_features import engineer
from src.monitoring.data_quality import validate_batch
from src.monitoring.perf_monitor import labelled_performance, score_batch


def data_quality_ok(df: pd.DataFrame) -> tuple[bool, list[str]]:
    r = validate_batch(df)
    return r["ok"], r["problems"]


def distribution_summary(ref: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in P.DRIFT_FEATURES:
        r, p = pd.to_numeric(ref[f], errors="coerce"), pd.to_numeric(prod[f], errors="coerce")
        rows.append(dict(feature=f, ref_median=r.median(), prod_median=p.median(), ref_mean=r.mean(), prod_mean=p.mean(),
                         ref_missing_pct=100 * r.isna().mean(), prod_missing_pct=100 * p.isna().mean()))
    return pd.DataFrame(rows).round(3)


def business_evidence(batch: pd.DataFrame, art: dict, labelled: dict | None) -> tuple[dict, bool]:
    """Cheap business-impact proxy: how much does the decision mix change?  And, with labels, is money being lost?"""
    p = score_batch(batch, art)
    shares = pd.Series([P.risk_level(v) for v in p]).value_counts(normalize=True).reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
    base_high = art["extra"].get("band_shares_test", {}).get("HIGH", 0.12)   # HIGH-band share on the test set at training time
    shift_high = float(shares["HIGH"] - base_high)
    info = dict(band_shares={k: round(float(v), 3) for k, v in shares.items()}, high_band_change_vs_training=round(shift_high, 3))
    high = abs(shift_high) >= 0.05
    if labelled:
        info["calibration_gap"] = labelled["calibration_gap"]
        high = high or labelled["calibration_gap"] <= -0.03       # model under-predicts defaults by 3+ percentage points
    return info, bool(high)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--ref", default=str(REFERENCE_PATH))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-effect", type=float, default=0.10)
    ap.add_argument("--labels-available", action="store_true",
                    help="Pretend 12 months have passed: use the TRUE default_12m column of the batch to measure real performance")
    ap.add_argument("--perf-drop", type=float, default=None, help="Manual override of the observed AUC drop")
    ap.add_argument("--business-impact-high", choices=["yes", "no"], default=None)
    a = ap.parse_args()

    ref, raw = pd.read_csv(a.ref), pd.read_csv(a.prod)
    prod = engineer(raw)                                   # same feature code as training / API
    art = load_artifact(CHAMPION_PATH)

    dq = validate_batch(raw)
    print(f"champion: {art['version']}   batch: {a.prod}  ({len(raw)} applications)\n")
    print("1. DATA QUALITY   ", "OK" if dq["ok"] else "FAILED", dq["checks"], f"income unit ratio={dq.get('income_unit_ratio')}")
    for pr in dq["problems"]:
        print("     !", pr)

    print("\n2. DRIFT (two-sample KS test, reference = training sample, production = this batch)")
    print("   H0: reference and production values come from the same distribution;  alpha =", a.alpha, " practical threshold D >=", a.min_effect)
    print(distribution_summary(ref, prod).to_string(index=False))
    rep = ks_drift_report(ref, prod, [f for f in P.DRIFT_FEATURES if f in prod.columns], alpha=a.alpha, min_effect=a.min_effect)
    print()
    print(rep.to_string(index=False))
    print()
    for _, r in rep.iterrows():
        print(" -", explain_row(r, a.alpha))

    labelled = None
    if a.labels_available and P.TARGET in raw.columns:
        labelled = labelled_performance(raw, art)
        print(f"\n3. PERFORMANCE on REAL outcomes: AUC {labelled['metrics']['roc_auc']} vs reference {labelled['reference_auc']} "
              f"(drop {labelled['perf_drop']}); mean predicted PD {labelled['mean_predicted_pd']} vs observed default rate "
              f"{labelled['observed_default_rate']}")
    else:
        print("\n3. PERFORMANCE   labels not available yet (defaults are known 12 months after disbursal) -> performance UNKNOWN")
    perf_drop = a.perf_drop if a.perf_drop is not None else (labelled["perf_drop"] if labelled else None)

    biz, biz_high = business_evidence(raw, art, labelled)
    print(f"\n4. BUSINESS IMPACT  decision mix {biz['band_shares']}  (HIGH-band share changed by {biz['high_band_change_vs_training']:+.3f} vs training)"
          + (f";  calibration gap {biz['calibration_gap']:+.3f}" if "calibration_gap" in biz else ""))
    impact = biz_high if a.business_impact_high is None else a.business_impact_high == "yes"
    print(f"   business impact judged {'HIGH' if impact else 'LOW'}")

    out = decide(rep, data_quality_ok=dq["ok"], perf_drop=perf_drop, labels_available=perf_drop is not None,
                 perf_drop_threshold=0.03, business_impact_high=impact if perf_drop is not None else None)
    print("\nDECISION:", out["decision"])
    for r in out["reasons"]:
        print("  *", r)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift_report.json").write_text(json.dumps(dict(batch=a.prod, table=rep.to_dict("records"), data_quality=dq,
                                                               performance=labelled, business=biz, decision=out), indent=2, default=str))
