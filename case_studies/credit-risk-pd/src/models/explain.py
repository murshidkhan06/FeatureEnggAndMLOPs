"""Explain the champion: which features drive its scores?  (Association, NOT causation.)

THREE LENSES
  1. Logistic baseline, STANDARDISED coefficients  - direction + strength for a linear model: +0.5 means "one standard
     deviation more of this feature raises the log-odds of default by 0.5, other features held fixed".
  2. PERMUTATION IMPORTANCE of the champion          - model-agnostic: shuffle ONE feature on held-out data and measure how much
     ROC-AUC drops.  Big drop = the model relies on that feature.  Computed on the TEST period (data the model never saw).
        tiny example: AUC 0.885 with the real column, 0.842 after shuffling credit_utilization -> importance 0.043.
  3. SHAP (OPTIONAL, `--shap`)                       - per-customer contributions that add up to the prediction.

CAVEATS TO SAY OUT LOUD
  * Correlated features SHARE importance: missed_payment_count and loan_payment_ratio carry almost the same information
    (corr -0.83), so shuffling one hurts less because the other still tells the story.  Importances are not additive.
  * Importance is model-specific and tied to the data period: it can change after retraining or under drift.
  * A feature near 0 may still be harmless noise (account_age): useful for the registry review - ask "why is it here?".
  * Importance != causality.  Reducing someone's utilisation does not necessarily lower their default risk.

Run:  python -m src.models.explain [--shap]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.common.artifact import load_artifact
from src.common.config import load_params
from src.common.paths import CHAMPION_PATH, REPORTS
from src.features.definitions import TARGET
from src.models.train import fit_model, load_snapshots

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true", help="OPTIONAL: also compute SHAP values (needs the shap package)")
    a = ap.parse_args()
    params = load_params()
    art = load_artifact(CHAMPION_PATH)
    feats = art["input_columns"]
    snap = load_snapshots()
    tr, te = snap[snap["split"] == "train"], snap[snap["split"] == "test"]
    report: dict = {"model": art["extra"].get("kind"), "features": feats}

    print("1) BASELINE logistic regression - standardised coefficients (train period)")
    lr = fit_model(tr, feats, "logistic", params)
    coefs = pd.Series(lr.named_steps["clf"].coef_[0][:len(feats)], index=feats).sort_values(key=abs, ascending=False)
    print(coefs.round(3).to_string())
    report["logistic_coefficients"] = coefs.round(4).to_dict()
    print("   note: the signs of correlated pairs (loan_payment_ratio / missed_payment_count) can look odd - they compete for the same signal.")

    print(f"\n2) PERMUTATION IMPORTANCE of the champion ({art['extra'].get('kind')}), ROC-AUC drop on the TEST period, 10 shuffles")
    r = permutation_importance(art["pipeline"], te[feats], te[TARGET], scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=1)
    imp = pd.DataFrame({"feature": feats, "auc_drop_mean": r.importances_mean.round(4), "auc_drop_std": r.importances_std.round(4)}) \
        .sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
    imp["share_of_total_%"] = (100 * imp["auc_drop_mean"].clip(lower=0) / imp["auc_drop_mean"].clip(lower=0).sum()).round(1)
    print(imp.to_string(index=False))
    report["permutation_importance"] = imp.to_dict("records")
    weak = imp[imp["auc_drop_mean"] < 0.002]["feature"].tolist()
    print(f"\n   Features with ~zero out-of-time importance: {weak or 'none'}  -> candidates to question in the next registry review.")
    print("   Correlated pair check: missed_payment_count vs loan_payment_ratio share their signal, so each looks weaker than it is alone.")

    if a.shap:
        print("\n3) SHAP (OPTIONAL)")
        try:
            import shap
            clf = art["pipeline"].named_steps["clf"]
            X = art["pipeline"].named_steps["prep"].transform(te[feats].head(400))
            sv = shap.TreeExplainer(clf).shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            m = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False).round(4)
            print("mean |SHAP value| (log-odds units) on 400 test customers:")
            print(m.to_string())
            report["shap_mean_abs"] = m.to_dict()
        except Exception as exc:                          # noqa: BLE001
            print(f"   SHAP unavailable for this model/package version ({type(exc).__name__}: {exc}); permutation importance above is the core result.")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "explainability.json").write_text(json.dumps(report, indent=2, default=str))
    from src.common.facts import save_fact
    save_fact("explain", report)
