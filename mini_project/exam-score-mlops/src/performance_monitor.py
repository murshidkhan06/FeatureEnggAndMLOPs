"""
src/performance_monitor.py

MODEL PERFORMANCE DECAY monitoring: compares the model's BASELINE performance (measured
once, at training time, on a held-out test set -- reports/evaluation_report.json) against
its RECENT PRODUCTION performance (measured on live predictions once their actual outcomes
become known -- the join of monitoring/predictions.csv and monitoring/labels.csv).

WHY this is a DIFFERENT thing from data drift (src/drift.py): data drift asks "do the
INPUTS still look like training data?" and needs no labels at all -- it can run on every
single request, in real time. Performance decay asks "is the model still ACCURATE?" and
fundamentally CANNOT be answered until the true outcome is known. For this project, that
means waiting for `POST /feedback` to arrive with an `actual_final_score` for a given
`request_id` (see src/monitoring.py's `record_actual_label` and its module docstring on
DELAYED LABELS). A production system might wait days between a prediction and knowing
whether it was right -- performance monitoring is inherently laggier than drift monitoring,
which is exactly why BOTH are needed, not just one.

HOW the comparison works:
    BASELINE  = metrics from reports/evaluation_report.json (the held-out test-set
                performance measured once, right after training -- e.g. "Baseline RMSE=7.2"
                from the course spec's worked example).
    RECENT    = metrics recomputed on whatever (prediction, actual_label) pairs have
                accumulated in the prediction store since then -- e.g. "Recent RMSE=11.5".
A model is flagged as DECAYED when the recent RMSE exceeds baseline RMSE by more than
`params.yaml: performance.max_rmse_increase`, AND there are at least
`performance.min_labeled_samples` labeled pairs to make that comparison meaningful (a
handful of unlucky predictions is not evidence of decay -- it's noise).

Run from the repo root:
    python -m src.performance_monitor
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent
MONITORING_DIR = REPO_ROOT / "monitoring"
PREDICTIONS_LOG = MONITORING_DIR / "predictions.csv"
LABELS_LOG = MONITORING_DIR / "labels.csv"
BASELINE_REPORT = REPO_ROOT / "reports" / "evaluation_report.json"
REPORT_PATH = REPO_ROOT / "reports" / "performance_report.json"


def load_params() -> dict:
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_baseline_metrics(path: Path = BASELINE_REPORT) -> Optional[Dict[str, float]]:
    """The model's own training-time held-out performance -- the yardstick everything else
    in this module is measured against. Written automatically by src/train.py."""
    if not path.exists():
        return None
    return json.loads(path.read_text())["metrics"]


def load_labeled_predictions(
    predictions_path: Path = PREDICTIONS_LOG,
    labels_path: Path = LABELS_LOG,
) -> pd.DataFrame:
    """Joins predictions.csv and labels.csv on request_id -- an INNER join, since a
    performance calculation is only possible for predictions that HAVE received their
    delayed label so far. Rows still waiting on a label (the common case, moment to moment,
    in a real system) are simply excluded until they get one."""
    if not predictions_path.exists() or not labels_path.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(predictions_path)
    predictions = predictions[predictions["status"] == "success"]
    labels = pd.read_csv(labels_path)

    joined = predictions.merge(
        labels, on="request_id", how="inner", suffixes=("", "_label")
    )
    return joined


def compute_recent_performance(joined: pd.DataFrame) -> Optional[Dict[str, float]]:
    if joined.empty:
        return None
    y_true = joined["actual_final_score"]
    y_pred = joined["predicted_final_score"]
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(joined) >= 2 else float("nan"),
        "n_labeled": int(len(joined)),
    }


def check_performance_decay(params: Optional[dict] = None) -> Dict[str, Any]:
    """The main entry point: loads baseline + recent metrics, decides whether performance
    has meaningfully decayed, and writes reports/performance_report.json."""
    params = params or load_params()
    perf_cfg = params["performance"]

    baseline = load_baseline_metrics()
    joined = load_labeled_predictions()
    recent = compute_recent_performance(joined)

    result: Dict[str, Any] = {
        "baseline": baseline,
        "recent": recent,
        "min_labeled_samples_required": perf_cfg["min_labeled_samples"],
        "max_rmse_increase_threshold": perf_cfg["max_rmse_increase"],
    }

    if baseline is None:
        result["status"] = "NO_BASELINE"
        result["decayed"] = False
        result["reason"] = "No reports/evaluation_report.json -- run `python -m src.train` first."
    elif recent is None or recent["n_labeled"] == 0:
        result["status"] = "NO_LABELS_YET"
        result["decayed"] = False
        result["reason"] = ("No labeled predictions yet -- send feedback via "
                             "`POST /feedback {request_id, actual_final_score}` once true "
                             "outcomes are known (see src/monitoring.record_actual_label).")
    elif recent["n_labeled"] < perf_cfg["min_labeled_samples"]:
        result["status"] = "INSUFFICIENT_SAMPLES"
        result["decayed"] = False
        result["reason"] = (
            f"Only {recent['n_labeled']} labeled predictions so far "
            f"(need >= {perf_cfg['min_labeled_samples']}) -- too few to trust a comparison."
        )
    else:
        rmse_increase = recent["rmse"] - baseline["rmse"]
        decayed = rmse_increase > perf_cfg["max_rmse_increase"]
        result["rmse_increase"] = round(rmse_increase, 3)
        result["status"] = "DECAYED" if decayed else "OK"
        result["decayed"] = decayed
        result["reason"] = (
            f"Recent RMSE ({recent['rmse']:.2f}) is {rmse_increase:+.2f} vs. baseline "
            f"({baseline['rmse']:.2f}); threshold is +{perf_cfg['max_rmse_increase']:.2f}."
        )

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    return result


def print_report(result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 66}\nModel performance-decay report\n{'=' * 66}")
    if result["baseline"]:
        b = result["baseline"]
        print(f"Baseline (training-time held-out test set): "
              f"MAE={b['mae']:.2f}  RMSE={b['rmse']:.2f}  R2={b['r2']:.3f}")
    if result["recent"]:
        r = result["recent"]
        print(f"Recent   ({r['n_labeled']} labeled production predictions): "
              f"MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  R2={r['r2']:.3f}")

    print(f"\nStatus: {result['status']}")
    print(f"Reason: {result['reason']}")
    if result["status"] == "DECAYED":
        print("\n[ALERT] Model performance has decayed beyond the configured threshold.")
        print("Reminder: performance decay is NOT the same as data drift -- this is measured "
              "from ACTUAL outcomes, and can happen even when src/drift.py reports no input "
              "drift at all (see src/concept_drift_demo.py). It is one of the required inputs "
              "to a retraining decision (src/retrain_decision.py), but not, by itself, a "
              "reason to retrain automatically.")


def main():
    parser = argparse.ArgumentParser(description="Check model performance decay against the training baseline.")
    _ = parser.parse_args()
    result = check_performance_decay()
    print_report(result)


if __name__ == "__main__":
    main()
