"""Train baselines + candidates on a CHRONOLOGICAL split, log everything to MLflow, save the CHALLENGER artifact.

The time-based split (never random!)
------------------------------------
      |------- TRAIN -------|--- VALID ---|--- TEST ---|======= PRODUCTION (future, unseen) =======>
      2023-01 .. 2024-06     2024-07..09   2024-10..12   2025-01 ...
  * TRAIN  : fit the preprocessing + model.
  * VALID  : choose between candidates (Ridge vs gradient boosting) and calibrate the prediction interval.
  * TEST   : touched ONCE at the end: the honest estimate of future accuracy.
  After choosing, the winner is RE-FITTED on TRAIN+VALID (more recent data = better), then scored on TEST.

Champion rule (from configs/params.yaml): lowest VALIDATION WAPE among the ML candidates whose |bias| <= 3%.
Why not RMSE?  A store manager thinks in "% of units sold that I got wrong" (WAPE) and "do I systematically over or
under stock" (bias).

Usage
-----
    python -m src.models.train --bootstrap-champion        # first model of the project
    python -m src.models.train --version v2                # a new CHALLENGER (champion untouched)
    python -m src.models.train --force-model ridge_log     # skip the champion rule (for experiments)
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src import project as P
from src.common.artifact import save_artifact
from src.common.mlflow_utils import setup, tracked_run
from src.common.params import load_params
from src.common.promotion import register_bootstrap
from src.common.paths import CHALLENGER_PATH, CHAMPION_PATH, DATA_RAW, MODELS
from src.features.build_features import training_frame
from src.features.registry import check_features_approved
from src.features.select_features import load_selected
from src.models import metrics as M
from src.models.baselines import BASELINES, baseline_forecast
from src.models.pipelines import MODEL_NAMES, build_pipeline


def time_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / valid / test using the dates in params.yaml.  Asserts the order (a cheap leakage guard)."""
    d = pd.to_datetime(df["date"])
    tr = df[d <= pd.Timestamp(split["train_end"])]
    va = df[(d > pd.Timestamp(split["train_end"])) & (d <= pd.Timestamp(split["valid_end"]))]
    te = df[(d > pd.Timestamp(split["valid_end"])) & (d <= pd.Timestamp(split["test_end"]))]
    assert tr["date"].max() < va["date"].min() and va["date"].max() < te["date"].min(), "splits must not overlap in time"
    return tr, va, te


def score(pipe, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(pipe.predict(frame[features]), 0, None)


def interval_quantiles(y: np.ndarray, f: np.ndarray, lo_q: float, hi_q: float) -> dict:
    """OPTIONAL prediction interval from RELATIVE validation residuals  r = (y - f) / f.

    Worked example: forecast 100, residual quantiles q10 = -0.22, q90 = +0.25  ->  interval [78, 125] units.
    Relative residuals make the band wide for big series and narrow for small ones."""
    rel = (np.asarray(y, float) - f) / np.maximum(f, 1.0)
    return dict(low=float(np.quantile(rel, lo_q)), high=float(np.quantile(rel, hi_q)), level=round(hi_q - lo_q, 2))


def coverage(y, f, q: dict) -> float:
    lo, hi = f * (1 + q["low"]), f * (1 + q["high"])
    return float(np.mean((np.asarray(y) >= lo) & (np.asarray(y) <= hi)))


def fit_and_select(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], params: dict,
                   force_model: str | None = None, verbose: bool = True) -> dict:
    """Fit each ML candidate on TRAIN, score on VALID, apply the champion rule.  Used by train.py AND retrain.py."""
    rule = params["champion_rule"]
    results = {}
    for name in MODEL_NAMES:
        pipe = build_pipeline(name, features, params).fit(train[features], train["sales"])
        f = score(pipe, valid, features)
        results[name] = dict(pipe=pipe, valid_pred=f, **M.report(valid["sales"], f, valid["rolling_mean_30"]))
        if verbose:
            r = results[name]
            print(f"  candidate {name:<10} valid WAPE={r['wape']:.4f}  bias={r['bias_pct']:+.2%}  MAE={r['mae']:.2f}")
    eligible = {k: v for k, v in results.items() if abs(v["bias_pct"]) <= rule["max_abs_bias_pct"]} or results
    best = force_model or min(eligible, key=lambda k: eligible[k]["wape"])
    return dict(results=results, best=best)


def refit_and_test(best: str, train_valid: pd.DataFrame, valid_pred: np.ndarray, valid: pd.DataFrame,
                   test: pd.DataFrame, features: list[str], params: dict) -> tuple:
    """Re-fit the winner on TRAIN+VALID and score TEST.  Returns (pipeline, metrics, interval, extras)."""
    pipe = build_pipeline(best, features, params).fit(train_valid[features], train_valid["sales"])
    f_test = score(pipe, test, features)
    q = interval_quantiles(valid["sales"].values, valid_pred, params["interval"]["low_quantile"], params["interval"]["high_quantile"])
    m = M.report(test["sales"], f_test, test["rolling_mean_30"])
    m["coverage"] = round(coverage(test["sales"].values, f_test, q), 4)
    return pipe, m, q, f_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--bootstrap-champion", action="store_true", help="also install as champion (first model)")
    ap.add_argument("--force-model", choices=MODEL_NAMES, default=None)
    a = ap.parse_args()

    params = load_params()
    features = load_selected()
    check_features_approved(features)                       # FEATURE GATE: refuse Rejected / unknown features
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    df = training_frame(raw)
    train, valid, test = time_split(df, params["split"])
    print(f"rows  train={len(train):,} ({train.date.min().date()}..{train.date.max().date()})  "
          f"valid={len(valid):,} ({valid.date.min().date()}..{valid.date.max().date()})  "
          f"test={len(test):,} ({test.date.min().date()}..{test.date.max().date()})")
    print(f"features ({len(features)}): {features}\n")

    setup(P.MODEL_NAME)
    common = dict(version=a.version, n_features=len(features), train_end=params["split"]["train_end"],
                  valid_end=params["split"]["valid_end"], test_end=params["split"]["test_end"])
    print("BASELINES (no learning) - validation and test")
    base_val, base_test = {}, {}
    for name in BASELINES:
        fv, ft = baseline_forecast(name, valid), baseline_forecast(name, test)
        base_val[name] = M.report(valid["sales"], fv, valid["rolling_mean_30"])
        base_test[name] = M.report(test["sales"], ft, test["rolling_mean_30"])
        print(f"  {name:<18} valid WAPE={base_val[name]['wape']:.4f} bias={base_val[name]['bias_pct']:+.2%} | "
              f"test WAPE={base_test[name]['wape']:.4f}")
        with tracked_run(f"baseline_{name}", dict(model=name, kind="baseline", **common)):
            for k, v in base_val[name].items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in base_test[name].items():
                mlflow.log_metric(f"test_{k}", v)

    print("\nCANDIDATES (fit on TRAIN, scored on VALID)")
    sel = fit_and_select(train, valid, features, params, a.force_model)
    best = sel["best"]
    for name, r in sel["results"].items():
        with tracked_run(f"candidate_{name}", dict(model=name, kind="candidate", **common)):
            for k, v in r.items():
                if k not in ("pipe", "valid_pred"):
                    mlflow.log_metric(f"valid_{k}", v)
    rule = params["champion_rule"]
    print(f"\nCHAMPION RULE: lowest validation WAPE with |bias| <= {rule['max_abs_bias_pct']:.0%}  ->  '{best}'")

    trv = pd.concat([train, valid])
    pipe, m, q, f_test = refit_and_test(best, trv, sel["results"][best]["valid_pred"], valid, test, features, params)
    skill = 1 - m["wape"] / base_test["seasonal_naive"]["wape"]
    m.update(val_wape=sel["results"][best]["wape"], val_bias_pct=sel["results"][best]["bias_pct"],
             skill_vs_seasonal_naive=round(skill, 4))
    print(f"\nFINAL '{best}' re-fitted on TRAIN+VALID ({len(trv):,} rows), scored ONCE on TEST:")
    print("  " + "  ".join(f"{k}={v}" for k, v in m.items()))
    print(f"  skill vs seasonal naive = 1 - {m['wape']:.4f}/{base_test['seasonal_naive']['wape']:.4f} = {skill:.1%} fewer errors")
    print(f"  80% interval (relative residual quantiles from VALID): [{q['low']:+.2f}, {q['high']:+.2f}] x forecast; "
          f"coverage on TEST = {m['coverage']:.1%}")

    with tracked_run(f"final_{best}_{a.version}", dict(model=best, kind="final", features=",".join(features), **common),
                     tags={"stage": "challenger"}) as run:
        for k, v in m.items():
            mlflow.log_metric(f"test_{k}" if k not in ("val_wape", "val_bias_pct") else k, v)
        example = test[features].head(3).copy()
        example[example.select_dtypes("integer").columns] = example.select_dtypes("integer").astype(float)
        # cloudpickle: HistGradientBoosting trees are not on MLflow's default (skops) trusted-types list
        mlflow.sklearn.log_model(pipe, name="model", input_example=example, serialization_format="cloudpickle",
                                 pip_requirements=["pandas", "numpy", "scikit-learn"])  # explicit list = fast, no auto-inference
        print(f"  MLflow run id: {run.info.run_id}")

    extra = dict(model_name=best, features=features, interval=q, train_rows=len(trv), trained_through=str(trv.date.max().date()),
                 split=params["split"], baselines_test={k: v["wape"] for k, v in base_test.items()},
                 candidates_valid={k: {kk: vv for kk, vv in r.items() if kk not in ("pipe", "valid_pred")}
                                   for k, r in sel["results"].items()})
    MODELS.mkdir(exist_ok=True)
    save_artifact(CHALLENGER_PATH, pipe, a.version, features, m, threshold=0.0, extra=extra)
    if a.bootstrap_champion or not CHAMPION_PATH.exists():
        save_artifact(CHAMPION_PATH, pipe, a.version, features, m, threshold=0.0, extra=extra)
        register_bootstrap(a.version, m)
        print(f"installed as champion ({a.version}) and recorded in models/registry.json")
    from src.common.facts import save_fact
    save_fact("train", dict(best=best, final_test=m, interval=q, baselines_valid=base_val, baselines_test=base_test,
                            candidates_valid=extra["candidates_valid"], features=features,
                            rows=dict(train=len(train), valid=len(valid), test=len(test))))


if __name__ == "__main__":
    main()
