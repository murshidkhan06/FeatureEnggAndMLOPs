"""Train baseline -> scorecard candidate -> gradient-boosting benchmark, log all to MLflow, save the CHALLENGER artifact.

Story (read the printed table top to bottom)
  1. BASELINE   logistic regression on the RAW approved columns only (no engineered features).
  2. CANDIDATE  logistic regression on the SELECTED engineered features (scorecard style).  <- the model we ship
  3. BENCHMARK  HistGradientBoosting on the same features (OPTIONAL-ish): is a more complex model worth it?

Model choice: keep the logistic regression unless boosting beats it by a MEANINGFUL margin (configs/params.yaml model.gb_margin
on the VALIDATION set).  Why?  Regulators and underwriters need reasons for every decision; a linear scorecard gives
exact reason codes, is stable, easy to audit and to explain.  The boosting model is kept as a CHALLENGER BENCHMARK
for monitoring ("is the simple model leaving a lot on the table?").

Training refuses any feature that is not Approved in the feature registry (check_features_approved).

Run:  python -m src.models.train [--version v1] [--bootstrap-champion]
"""
from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src import project as P
from src.common.artifact import save_artifact
from src.common.mlflow_utils import setup, tracked_run
from src.common.promotion import register_bootstrap
from src.common.params import load_params
from src.common.paths import CHALLENGER_PATH, CHAMPION_PATH, CONFIGS, DATA_PROCESSED, MODELS, REPORTS
from src.features.definitions import RAW_ELIGIBLE
from src.features.registry import check_features_approved, load_selected
from src.models.metrics import credit_metrics
from src.models.pipeline import build_pipeline, fit_score, load_splits  # noqa: F401  (re-exported for other modules)

# raw LEVEL columns a "no feature engineering" analyst would use: no ratios (debt_to_income_ratio is a ratio); location excluded by governance
BASELINE_FEATURES = [c for c in RAW_ELIGIBLE if c not in ("location", "debt_to_income_ratio")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--C", type=float, default=None, help="L2 strength of the scorecard (default: params.yaml model.C)")
    ap.add_argument("--bootstrap-champion", action="store_true", help="also install as champion (first model)")
    a = ap.parse_args()
    prm = load_params()
    C = a.C or prm["model"]["C"]
    train, valid, test = load_splits()
    selected = load_selected()
    check_features_approved(selected)          # governance guard: refuse unapproved / sensitive / leaky features
    check_features_approved(BASELINE_FEATURES)
    print(f"train={len(train)} valid={len(valid)} test={len(test)}   selected features ({len(selected)}): {selected}\n")

    setup(P.MODEL_NAME)
    runs = {}
    for name, feats, model in [("baseline_logreg_raw", BASELINE_FEATURES, "logreg"),
                               ("candidate_logreg_selected", selected, "logreg"),
                               ("benchmark_hgb_selected", selected, "hgb")]:
        params = dict(model=model, C=C if model == "logreg" else "n/a", n_features=len(feats), n_train=len(train),
                      version=a.version, split="time-based 70/15/15", features=",".join(feats))
        with tracked_run(f"{name}_{a.version}", params, tags={"stage": "development", "project": "credit-scoring"}):
            pipe, mv, mt = fit_score(feats, model, C, train, valid, test)
            for k, v in mv.items():
                if isinstance(v, float):
                    mlflow.log_metric(f"valid_{k}", v)
            for k, v in mt.items():
                if isinstance(v, float):
                    mlflow.log_metric(f"test_{k}", v)
            mlflow.log_artifact(str(CONFIGS / "selected_features.yaml"))     # lineage: which features, which registry
            mlflow.log_artifact(str(CONFIGS / "feature_registry.yaml"))
            # cloudpickle: works for every pipeline in every MLflow version (MLflow 3 defaults to 'skops', which
            # refuses numpy dtypes stored inside the OrdinalEncoder).  Only load models you trained yourself.
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        runs[name] = dict(pipeline=pipe, valid=mv, test=mt, features=feats)

    rows = [dict(model=n, n_features=len(r["features"]), **{f"valid_{k}": r["valid"][k] for k in ("roc_auc", "ks_stat")},
                 **{f"test_{k}": r["test"][k] for k in ("roc_auc", "gini", "ks_stat", "pr_auc", "recall_at_80_approval")})
            for n, r in runs.items()]
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))

    cand, bench, base = runs["candidate_logreg_selected"], runs["benchmark_hgb_selected"], runs["baseline_logreg_raw"]
    gain_feat = cand["valid"]["roc_auc"] - base["valid"]["roc_auc"]
    gain_feat_test = cand["test"]["roc_auc"] - base["test"]["roc_auc"]
    gain_gb = bench["valid"]["roc_auc"] - cand["valid"]["roc_auc"]
    margin = prm["model"]["gb_margin"]
    print(f"\nfeature engineering gain (candidate - baseline) AUC     : valid {gain_feat:+.4f} | test {gain_feat_test:+.4f}"
          "   (~200 defaulters per split: differences of +-0.02 are within noise)")
    print(f"boosting gain over the scorecard (valid AUC, hgb - logreg)  : {gain_gb:+.4f}   (needed: >= {margin})")
    if gain_gb >= margin:
        print("-> boosting wins by a meaningful margin.  Even so, the shipped model stays the scorecard unless Risk + Compliance "
              "approve a non-linear model with SHAP-based reasons (policy decision).")
    else:
        print("-> boosting does NOT win by a meaningful margin: keep the interpretable logistic-regression scorecard.")

    pipe = cand["pipeline"]
    final_metrics = {k: v for k, v in cand["test"].items() if k not in ("n", "default_rate")}
    p_test = pipe.predict_proba(test[selected])[:, 1]
    band_shares = pd.Series([P.risk_level(v) for v in p_test]).value_counts(normalize=True).reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
    extra = dict(model_family="logistic_regression_scorecard", features=selected, C=C,
                 cutoffs=dict(refer=P.REFER_CUTOFF, decline=P.DECLINE_CUTOFF),
                 valid_metrics=cand["valid"], baseline_test=base["test"], benchmark_hgb_test=bench["test"],
                 train_period=[train.application_date.min(), train.application_date.max()], n_train=len(train),
                 feature_registry_version="1.0", band_shares_test={k: round(float(v), 4) for k, v in band_shares.items()})
    MODELS.mkdir(exist_ok=True)
    save_artifact(CHALLENGER_PATH, pipe, a.version, selected, final_metrics, threshold=P.DECLINE_CUTOFF, extra=extra)
    joblib.dump(dict(pipeline=bench["pipeline"], features=selected, version=a.version), MODELS / "benchmark_hgb.joblib")
    if a.bootstrap_champion or not CHAMPION_PATH.exists():
        save_artifact(CHAMPION_PATH, pipe, a.version, selected, final_metrics, threshold=P.DECLINE_CUTOFF, extra=extra)
        register_bootstrap(a.version, final_metrics)         # first champion goes into models/registry.json so a later rollback can restore it
        print(f"\ninstalled {a.version} as champion  (test metrics: {final_metrics}) and recorded it in models/registry.json")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(
        dict(version=a.version, test=final_metrics, comparison=table.to_dict("records"),
             feature_engineering_gain_valid_auc=round(gain_feat, 4), feature_engineering_gain_test_auc=round(gain_feat_test, 4), boosting_gain_valid_auc=round(gain_gb, 4)), indent=2))


if __name__ == "__main__":
    main()
