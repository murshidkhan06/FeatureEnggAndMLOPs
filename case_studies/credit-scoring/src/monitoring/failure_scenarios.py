"""Reproduce the three classic production failures of a credit model - and show how each one is DETECTED.

(a) Approval rate suddenly JUMPS      cause: an upstream system starts sending annual amounts in the monthly_income field (unit change)
                                      detect: data-quality unit check + KS on debt_to_income_ratio + decision mix; action: FIX THE PIPELINE, do not retrain
(b) AUC FELL after a "new feature"    cause: a model trained with a leaky column (interest_rate_offered) is deployed; in production the column does not exist
                                      detect: labelled performance (late!) - prevention: registry check + leakage tests
(c) 422 SPIKE after a schema change   cause: upstream renames a field / changes a category spelling; the API contract rejects the requests
                                      detect: api_requests_total{status="422"} ratio in Prometheus -> alert

Run:  python -m src.monitoring.failure_scenarios
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from src import project as P
from src.api.app import app
from src.common.artifact import load_artifact
from src.common.drift import ks_drift_report
from src.common.paths import CHAMPION_PATH, DATA_PRODUCTION, REFERENCE_PATH, REPORTS
from src.common.retrain_decision import decide
from src.features.build_features import engineer
from src.models.pipeline import build_pipeline, load_splits
from src.monitoring.data_quality import validate_batch
from src.monitoring.perf_monitor import score_batch


def scenario_a(art: dict) -> dict:
    print("=" * 96 + "\n(a) APPROVAL RATE JUMPS - upstream unit change (monthly_income now carries ANNUAL amounts)\n" + "=" * 96)
    ok = pd.read_csv(DATA_PRODUCTION / "batch_normal.csv")
    bug = ok.copy()
    bug["monthly_income"] = (bug["monthly_income"] * 12).round(-2)
    ref = pd.read_csv(REFERENCE_PATH)
    out = {}
    for name, df in (("normal batch", ok), ("batch after the unit change", bug)):
        p = score_batch(df, art)
        bands = pd.Series([P.risk_level(v) for v in p]).value_counts(normalize=True).reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        dq = validate_batch(df)
        out[name] = dict(approval_rate=round(float((p < art["threshold"]).mean()), 4), low_band=round(float(bands["LOW"]), 4),
                         high_band=round(float(bands["HIGH"]), 4), mean_pd=round(float(p.mean()), 4), data_quality_ok=dq["ok"])
        print(f"  {name:30s} not-declined {out[name]['approval_rate']:.1%} | LOW band {bands['LOW']:.1%} | HIGH band {bands['HIGH']:.1%} | mean PD {p.mean():.3f} | data quality {'OK' if dq['ok'] else 'FAILED'}")
    dqb = validate_batch(bug)
    print("  data-quality problems:", dqb["problems"])
    rep = ks_drift_report(ref, engineer(bug), P.DRIFT_FEATURES)
    print("  KS drift:", {r.feature: f"D={r.ks_statistic:.2f}" for r in rep.itertuples() if r.status == "drift"})
    d = decide(rep, data_quality_ok=dqb["ok"])
    print(f"  LADDER DECISION: {d['decision']}  -> {d['reasons'][0]}")
    out["decision"] = d["decision"]
    return out


def scenario_b(art: dict) -> dict:
    print("\n" + "=" * 96 + "\n(b) AUC FELL AFTER A NEW FEATURE WAS ADDED - the leaky column vanishes in production\n" + "=" * 96)
    train, valid, test = load_splits()
    feats = art["input_columns"] + ["interest_rate_offered"]
    leaky = build_pipeline(feats).fit(train[feats], train[P.TARGET])
    offline = roc_auc_score(test[P.TARGET], leaky.predict_proba(test[feats])[:, 1])
    prod = pd.read_csv(DATA_PRODUCTION / "batch_normal.csv")
    fr = engineer(prod)
    fr["interest_rate_offered"] = np.nan                      # in production the rate has not been set yet
    prod_auc = roc_auc_score(prod[P.TARGET], leaky.predict_proba(fr[feats])[:, 1])
    champ_auc = roc_auc_score(prod[P.TARGET], score_batch(prod, art))
    print(f"  leaky model  : offline test AUC {offline:.3f}  ->  production AUC {prod_auc:.3f}   (the 'new feature' was a mistake)")
    print(f"  honest champion on the same production batch: AUC {champ_auc:.3f}")
    print("  Prevention: feature registry blocks it at training time (check_features_approved) + leakage tests; detection after the fact needs LABELS (12 months!).")
    return dict(offline_auc=round(offline, 4), production_auc=round(prod_auc, 4), honest_champion_production_auc=round(champ_auc, 4))


def scenario_c() -> dict:
    print("\n" + "=" * 96 + "\n(c) API 422 SPIKE AFTER A SCHEMA CHANGE - upstream renames monthly_debt and capitalises employment_type\n" + "=" * 96)
    batch = pd.read_csv(DATA_PRODUCTION / "batch_normal.csv").head(60)
    fields = list(P.PredictRequest.model_fields)

    def payload(row, changed: bool) -> dict:
        d = {k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k])) for k in fields}
        d = {k: v for k, v in d.items() if v is not None}
        if changed:
            d["monthly_debt_inr"] = d.pop("monthly_debt")              # renamed field
            d["employment_type"] = str(d["employment_type"]).capitalize()  # 'Salaried' is not a valid value
        return d

    res = {}
    with TestClient(app) as c:
        for label, changed in (("before the change", False), ("after the change", True)):
            codes = [c.post("/predict", json=payload(r, changed)).status_code for _, r in batch.iterrows()]
            res[label] = dict(ok=codes.count(200), rejected_422=codes.count(422), share_422=round(codes.count(422) / len(codes), 3))
            print(f"  {label:18s}: HTTP 200 x {codes.count(200)}, HTTP 422 x {codes.count(422)}  ({codes.count(422) / len(codes):.0%} rejected)")
        err = c.post("/predict", json=payload(batch.iloc[0], True)).json()
        print("  what the caller sees:", json.dumps(err["detail"][:2]))
        metrics = [ln for ln in c.get("/metrics").text.splitlines() if ln.startswith('api_requests_total{endpoint="/predict"')]
    print("  Prometheus counters:")
    for ln in metrics:
        print("    ", ln)
    print("  Alert rule (monitoring/alert_rules.yml): sum(rate(api_requests_total{status='422'}[5m])) / sum(rate(api_requests_total[5m])) > 0.05")
    print("  Fix: version the contract (v1/v2 endpoints), contract tests in the upstream CI, and tolerate-or-reject decisions made explicitly.")
    return res


def main() -> None:
    art = load_artifact(CHAMPION_PATH)
    out = dict(a=scenario_a(art), b=scenario_b(art), c=scenario_c())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "failure_scenarios.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
