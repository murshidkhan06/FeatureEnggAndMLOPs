"""Explainability for the scorecard: three global views + one local view (reason codes).

GLOBAL ("which features matter overall?")
  1. Standardized coefficients   logistic regression after StandardScaler: coefficient = change in log-odds of default per
                                 +1 standard deviation of the feature.  Sign = direction, size = strength (comparable across features).
  2. Permutation importance      shuffle ONE feature on the test set and see how much AUC drops.  Model-agnostic; uses the
                                 real metric.  Tiny example: AUC 0.77 -> 0.70 after shuffling DTI means DTI matters (0.07).
  3. SHAP (OPTIONAL)             Shapley-value attributions.  Used on the gradient-boosting BENCHMARK; for a linear model SHAP
                                 gives the same story as coefficient x value, so it is not needed for the scorecard.
LOCAL ("why THIS applicant?")
  4. Reason codes                top-3 adverse factors, see src/models/reason_codes.py.

Caveats to remember: correlated features SHARE importance (permuting one while its twin remains hides its value);
importance is about the MODEL, not about causes (association, not causation).

Run:  python -m src.models.explain
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import project as P
from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, MODELS, REPORTS
from src.models.pipeline import encoded_to_feature
from src.models.reason_codes import reason_codes, standardized_coefficients
from src.models.pipeline import load_splits


def shap_summary(bench: dict, X: pd.DataFrame) -> pd.DataFrame | None:
    """OPTIONAL: mean |SHAP| per original feature for the gradient-boosting benchmark."""
    try:
        import shap
    except ImportError:
        print("   (shap is not installed: skipping the OPTIONAL SHAP section)")
        return None
    pipe, feats = bench["pipeline"], bench["features"]
    Z = pipe[:-1].transform(X[feats])
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    try:
        values = shap.TreeExplainer(pipe[-1]).shap_values(Z)
    except Exception:                                                     # noqa: BLE001  (older/newer shap versions differ)
        f = lambda z: pipe[-1].predict_proba(z)[:, 1]                     # noqa: E731
        values = shap.Explainer(f, shap.maskers.Independent(Z[:100]))(Z[:200]).values
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    imp = pd.DataFrame(dict(owner=[encoded_to_feature(n, feats) for n in names], v=np.abs(values).mean(axis=0)))
    return imp.groupby("owner").v.sum().sort_values(ascending=False).rename("mean_abs_shap").reset_index().rename(columns={"owner": "feature"})


def main() -> None:
    art = load_artifact(CHAMPION_PATH)
    pipe, feats = art["pipeline"], art["input_columns"]
    _, _, test = load_splits()
    out: dict = {}

    print(f"CHAMPION {art['version']} - logistic regression scorecard on {len(feats)} features\n")
    sc = standardized_coefficients(pipe, feats)
    print("1. STANDARDIZED COEFFICIENTS (log-odds of default per +1 std; + = riskier)")
    print(sc[["feature", "encoded", "coef"]].round(3).to_string(index=False))
    out["standardized_coefficients"] = sc[["feature", "encoded", "coef"]].round(4).to_dict("records")

    pi = permutation_importance(pipe, test[feats], test[P.TARGET], scoring="roc_auc", n_repeats=10, random_state=0)
    pit = pd.DataFrame(dict(feature=feats, auc_drop=pi.importances_mean, std=pi.importances_std)).sort_values("auc_drop", ascending=False)
    print("\n2. PERMUTATION IMPORTANCE on the TEST set (AUC drop when the feature is shuffled, 10 repeats)")
    print(pit.round(4).to_string(index=False))
    out["permutation_importance"] = pit.round(4).to_dict("records")
    print("   Note: correlated features share importance (e.g. DTI and debt_service_ratio); read groups, not single rows.")

    bench_path = MODELS / "benchmark_hgb.joblib"
    print("\n3. SHAP on the gradient-boosting benchmark (OPTIONAL)")
    if bench_path.exists():
        sh = shap_summary(joblib.load(bench_path), test.sample(300, random_state=0))
        if sh is not None:
            print(sh.round(4).to_string(index=False))
            out["shap_hgb_benchmark"] = sh.round(4).to_dict("records")
    else:
        print("   benchmark model not found (run training first)")

    print("\n4. REASON CODES (top-3 adverse factors) for three test applicants")
    p = pipe.predict_proba(test[feats])[:, 1]
    picks = {"LOW": np.argsort(p)[len(p) // 4], "MEDIUM": np.argsort(p)[int(len(p) * 0.78)], "HIGH": np.argsort(p)[int(len(p) * 0.97)]}
    out["reason_code_examples"] = []
    for band, i in picks.items():
        row = test.iloc[[i]]
        codes = reason_codes(pipe, row, feats, 3)[0]
        print(f"\n   {band}-ish applicant (PD={p[i]:.3f}, band={P.risk_level(p[i])}, actually defaulted={int(row[P.TARGET].iloc[0])})")
        for c in codes:
            print(f"     +{c['contribution']:.2f}  {c['reason']}")
        out["reason_code_examples"].append(dict(pd=round(float(p[i]), 4), band=P.risk_level(p[i]), codes=codes))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "explain.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
