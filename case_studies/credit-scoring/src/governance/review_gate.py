"""GO / NO-GO gate before a model may be promoted to production (the last automated step before human committees).

The gate FAILS (exit code 1) if any of these is true:
  1. a model feature is unapproved in the registry (unknown, Rejected, leakage, ...)
  2. a sensitive / restricted / proxy-risk feature (gender, marital_status, age, fine location) is in the model
  3. adverse impact ratio (gender) is below 0.80   (four-fifths rule, configs/params.yaml fairness.min_adverse_impact_ratio)
  4. the model card / review pack is missing
  5. test ROC-AUC is below the minimum in configs/quality_gate.json
  6. reason codes cannot be produced (a model without per-applicant reasons cannot be used for lending decisions here)
A PASS only means "ready for the HUMAN review steps"; it never replaces them (docs/REGULATORY_REVIEW_FLOW.md).

Run:  python -m src.governance.review_gate                    # check the champion
      python -m src.governance.review_gate --demo-bad-model    # a deliberately bad model: watch every check fail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH, CONFIGS, REPORTS
from src.features.registry import FeatureNotApproved, check_features_approved, load_registry
from src.governance.fairness_audit import run_audit
from src.models.pipeline import build_pipeline, load_splits
from src.models.reason_codes import reason_codes

CARD_PATH = REPORTS / "model_review_pack.md"
REQUIRED_SECTIONS = ["Model purpose", "Data lineage", "Feature approval", "Rejected features", "Performance", "Stability", "Fairness",
                     "Explainability", "Limitations", "Monitoring plan", "Sign-off"]
BLOCKED_SENSITIVITY = {"sensitive_attribute", "restricted_attribute", "proxy_risk"}


def check_reason_codes(art: dict, sample: pd.DataFrame) -> tuple[bool, str]:
    clf = art["pipeline"].named_steps.get("clf")
    if not hasattr(clf, "coef_"):
        return False, f"{type(clf).__name__} has no coefficients: cannot produce linear reason codes"
    try:
        codes = reason_codes(art["pipeline"], sample, art["input_columns"], top_k=3)
    except Exception as exc:                                         # noqa: BLE001
        return False, f"reason code computation failed: {exc}"
    ok = all(len(c) >= 1 for c in codes) and max(len(c) for c in codes) <= 3
    return ok, f"reason codes produced for {len(codes)} sample applicants (max 3 each)"


def run_gate(art: dict, card_path: Path = CARD_PATH, registry: dict | None = None) -> list[dict]:
    prm = load_params()
    reg = registry or load_registry()
    gate = json.loads((CONFIGS / "quality_gate.json").read_text())
    feats = art["input_columns"]
    results = []

    def add(name, ok, detail):
        results.append(dict(check=name, passed=bool(ok), detail=detail))

    try:
        check_features_approved(feats, reg)
        add("1 all features approved in registry", True, f"{len(feats)} features approved")
    except FeatureNotApproved:
        bad = [f"{f} ({reg[f]['decision_category']})" if f in reg else f"{f} (unknown)" for f in feats if f not in reg or reg[f]["status"] != "Approved"]
        add("1 all features approved in registry", False, f"not approved: {', '.join(bad)}")
    blocked = [f for f in feats if f not in reg or reg[f].get("sensitivity") in BLOCKED_SENSITIVITY or reg[f].get("decision_category") in ("SENSITIVE_POLICY", "REQUIRES_GOVERNANCE")]
    add("2 no sensitive / restricted / proxy feature in model", not blocked, "none" if not blocked else f"found: {blocked}")
    try:
        aud = run_audit(art)
        air = aud["gender_summary"]["air"]
        add(f"3 adverse impact ratio ({aud['gate_attribute']}) >= {aud['min_air']}", air >= aud["min_air"], f"AIR = {air}")
    except Exception as exc:                                         # noqa: BLE001
        add("3 adverse impact ratio", False, f"audit could not run: {exc}")
    text = card_path.read_text(encoding="utf-8") if card_path.exists() else ""
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in text.lower()]
    add("4 model card / review pack present and complete", card_path.exists() and not missing,
        f"{card_path.name} found, all {len(REQUIRED_SECTIONS)} sections" if card_path.exists() and not missing else
        (f"{card_path.name} NOT FOUND" if not card_path.exists() else f"missing sections: {missing}"))
    auc, min_auc = art["metrics"].get("roc_auc", 0), gate["min"]["roc_auc"]
    add(f"5 test ROC-AUC >= {min_auc}", auc >= min_auc, f"AUC = {auc}")
    _, _, test = load_splits()
    ok, detail = check_reason_codes(art, test.head(20)) if all(f in test.columns for f in feats) else (False, "features missing in data")
    add("6 reason codes available", ok, detail)
    return results


def print_gate(results: list[dict]) -> bool:
    for r in results:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']:62s} {r['detail']}")
    ok = all(r["passed"] for r in results)
    print("\nGATE:", "GO - ready for the human review steps (independent validation, compliance, model risk committee)"
          if ok else "NO-GO - do not promote; fix the failed checks")
    return ok


def bad_model_artifact() -> dict:
    """A deliberately bad model: leaky column + sensitive columns + black-box learner."""
    train, valid, test = load_splits()
    feats = load_artifact(CHAMPION_PATH)["input_columns"] + ["interest_rate_offered", "gender", "age", "location"]
    pipe = build_pipeline(feats, model="hgb").fit(train[feats], train[P.TARGET])
    from src.models.metrics import credit_metrics
    m = credit_metrics(test[P.TARGET], pipe.predict_proba(test[feats])[:, 1])
    return dict(pipeline=pipe, input_columns=feats, metrics=m, threshold=P.DECLINE_CUTOFF, version="BAD-DEMO", extra={})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-bad-model", action="store_true")
    a = ap.parse_args()
    if a.demo_bad_model:
        print("REVIEW GATE on a deliberately BAD model (leaky + sensitive features, black-box learner, no model card)")
        art = bad_model_artifact()
        print(f"  its test AUC is {art['metrics']['roc_auc']}  <- 'great' accuracy does not make a model acceptable\n")
        ok = print_gate(run_gate(art, card_path=REPORTS / "model_review_pack_BAD_DEMO.md"))
        sys.exit(1 if not ok else 0)
    art = load_artifact(CHAMPION_PATH)
    print(f"REVIEW GATE for champion {art['version']}  (features: {art['input_columns']})\n")
    ok = print_gate(run_gate(art))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
