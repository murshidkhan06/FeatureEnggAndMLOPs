"""Train the baseline (logistic regression) and the candidate (gradient boosting), log both to MLflow, save the winner.

WHY this order: always build the SIMPLEST sensible model first (the baseline).  The complex model must EARN its place:
we keep logistic regression unless gradient boosting beats it on the VALIDATION period by at least `min_gain` ROC-AUC.
(Parsimony: a scorecard-like model is easier to explain to regulators and cheaper to run.)

TIME-BASED SPLIT (never random: customers repeat across months and the future must not leak into the past)
    train   as_of 2021-06-30 .. 2022-03-31   fit all preprocessing + model here
    purge   2022-04-30 .. 2022-08-31         labels of these rows were not yet known when the model was built -> unused
    val     2022-09-30 .. 2022-11-30         choose between baseline and candidate
    test    2022-12-31 .. 2023-02-28         report ONCE; never used for any decision
    later months (production) are used only by monitoring and retraining.

THE FEATURE GATE: before fitting, check_features_approved() verifies every feature is Approved + Production in the registry.
A leaky or unregistered feature stops training with a clear error.

Run:  python -m src.models.train --bootstrap-champion
"""
from __future__ import annotations

import argparse
import hashlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from src import project as P
from src.common.artifact import save_artifact
from src.common.config import load_params
from src.common.facts import save_fact
from src.common.promotion import register_bootstrap
from src.common.mlflow_utils import setup, tracked_run
from src.common.paths import (CHALLENGER_PATH, CHAMPION_PATH, FEATURE_REGISTRY_PATH, MODELS, SELECTED_FEATURES_PATH,
                              SNAPSHOTS_PATH)
from src.features.build_features import add_derived_features
from src.features.definitions import MODEL_FEATURES, TARGET
from src.governance.feature_lifecycle import check_features_approved
from src.models.evaluate import compute_metrics, evaluate_pipeline
from src.models.pipelines import MAKERS

MIN_GAIN = 0.005       # complex model must beat the simple one by this much ROC-AUC on validation


def load_snapshots() -> pd.DataFrame:
    """Snapshot table with the 3 derived features added (same function the API uses)."""
    return add_derived_features(pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"]))


def selected_features() -> list[str]:
    """Features chosen by the selection step; falls back to all approved features if that step has not been run."""
    if SELECTED_FEATURES_PATH.exists():
        return yaml.safe_load(SELECTED_FEATURES_PATH.read_text(encoding="utf-8"))["selected"]
    print("WARNING: configs/selected_features.yaml missing -> using all approved features (run select_features first)")
    return list(MODEL_FEATURES)


def fit_model(train_df: pd.DataFrame, features: list[str], kind: str, params: dict | None = None):
    """Fit ONE pipeline (imputer/scaler/model all learn from train_df only).  kind = 'logistic' | 'hgb'."""
    params = params or load_params()
    check_features_approved(features)                         # <- governance gate
    return MAKERS[kind](features, params["model"][kind]).fit(train_df[features], train_df[TARGET])


def data_fingerprint() -> str:
    """Short hash of snapshots.csv so every MLflow run records WHICH data it saw (poor man's dataset version)."""
    return hashlib.md5(SNAPSHOTS_PATH.read_bytes()).hexdigest()[:10]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--bootstrap-champion", action="store_true", help="also install the result as champion (first model)")
    a = ap.parse_args()

    params = load_params()
    top = params["model"]["top_pct"]
    snap = load_snapshots()
    features = selected_features()
    tr, va, te = (snap[snap["split"] == s] for s in ("train", "val", "test"))
    print(f"features ({len(features)}): {features}")
    print(f"rows  train={len(tr):,} val={len(va):,} test={len(te):,}   default rate  {tr[TARGET].mean():.2%} / {va[TARGET].mean():.2%} / {te[TARGET].mean():.2%}")

    setup(P.MODEL_NAME)
    results: dict = {}
    for kind in ("logistic", "hgb"):
        run_params = dict(model=kind, version=a.version, n_features=len(features), features=",".join(features),
                          train_period=f"{tr['as_of_date'].min().date()}..{tr['as_of_date'].max().date()}",
                          val_period=f"{va['as_of_date'].min().date()}..{va['as_of_date'].max().date()}",
                          test_period=f"{te['as_of_date'].min().date()}..{te['as_of_date'].max().date()}",
                          n_train=len(tr), data_hash=data_fingerprint(), **{f"hp_{k}": v for k, v in params["model"][kind].items()})
        with tracked_run(f"{kind}_{a.version}", run_params, tags={"stage": "baseline" if kind == "logistic" else "candidate"}):
            pipe = fit_model(tr, features, kind, params)
            m_tr, _ = evaluate_pipeline(pipe, tr, features, TARGET, top)
            m_va, s_va = evaluate_pipeline(pipe, va, features, TARGET, top)
            m_te, _ = evaluate_pipeline(pipe, te, features, TARGET, top)
            for split, m in (("train", m_tr), ("val", m_va), ("test", m_te)):
                for k in ("roc_auc", "pr_auc", "recall_at_10pct", "lift_at_10pct", "ks"):
                    mlflow.log_metric(f"{split}_{k}", m[k])
            mlflow.log_artifact(str(SELECTED_FEATURES_PATH))
            mlflow.log_artifact(str(FEATURE_REGISTRY_PATH))
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle",
                                     pip_requirements=["scikit-learn", "pandas", "numpy"])   # explicit list = fast + reproducible
        results[kind] = dict(pipe=pipe, train=m_tr, val=m_va, test=m_te, val_scores=s_va)

    print(f"\n{'model':<19}{'period':<7}{'ROC-AUC':>9}{'PR-AUC':>9}{'recall@10%':>12}{'lift@10%':>10}{'KS':>8}")
    for kind, label in (("logistic", "BASELINE  logistic"), ("hgb", "CANDIDATE hist-GB")):
        for split in ("train", "val", "test"):
            m = results[kind][split]
            print(f"{label if split == 'train' else '':<19}{split:<7}{m['roc_auc']:>9.4f}{m['pr_auc']:>9.4f}"
                  f"{m['recall_at_10pct']:>12.4f}{m['lift_at_10pct']:>10.2f}{m['ks']:>8.4f}")
    gain = results["hgb"]["val"]["roc_auc"] - results["logistic"]["val"]["roc_auc"]
    winner = "hgb" if gain >= MIN_GAIN else "logistic"
    print(f"\nValidation ROC-AUC gain of the candidate over the baseline: {gain:+.4f}  (rule: keep the simple model unless gain >= {MIN_GAIN})")
    print(f"WINNER -> {winner}.  (train > val > test is normal: the model has seen the train customers; trust val/test.)")

    best = results[winner]
    threshold = float(np.quantile(best["val_scores"], 1 - top))     # contact threshold: top 10 % of VALIDATION scores
    metrics = {k: best["test"][k] for k in ("roc_auc", "pr_auc", "recall_at_10pct", "lift_at_10pct", "ks")}
    extra = dict(kind=winner, val_metrics=best["val"], test_base_rate=best["test"]["base_rate"], test_mean_pd=best["test"]["mean_score"], features=features,
                 baseline_test=results["logistic"]["test"], data_hash=data_fingerprint(),
                 train_period=f"{tr['as_of_date'].min().date()}..{tr['as_of_date'].max().date()}", threshold_rule=f"top {top:.0%} of validation scores")
    MODELS.mkdir(exist_ok=True)
    save_fact("train", dict(features=features, winner=winner, val_gain=round(gain, 4), min_gain=MIN_GAIN, contact_threshold=round(threshold, 4),
                            rows=dict(train=len(tr), val=len(va), test=len(te)),
                            models={k: dict(train=results[k]["train"], val=results[k]["val"], test=results[k]["test"]) for k in results}, final_test_metrics=metrics))
    save_artifact(CHALLENGER_PATH, best["pipe"], a.version, features, metrics, threshold=threshold, extra=extra)
    print(f"saved models/challenger.joblib  version={a.version}  test metrics={metrics}  contact threshold={threshold:.4f}")
    if a.bootstrap_champion or not CHAMPION_PATH.exists():
        save_artifact(CHAMPION_PATH, best["pipe"], a.version, features, metrics, threshold=threshold, extra=extra)
        register_bootstrap(a.version, metrics)                     # so that a later rollback has a version to return to
        print("installed as champion (models/champion.joblib) and recorded in models/registry.json")
