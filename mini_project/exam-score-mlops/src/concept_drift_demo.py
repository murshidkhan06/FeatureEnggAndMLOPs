"""
src/concept_drift_demo.py

CONCEPT DRIFT demonstration, using the SAME Exam Score problem as everything else in this
project.

WHY a separate module from src/drift.py? Because concept drift is a fundamentally
different phenomenon from data drift, and needs a fundamentally different kind of evidence
to detect:

    DATA DRIFT:     P(X) changes.        Detected by comparing FEATURE distributions
                                          (src/drift.py's KS test / chi-square test).
    CONCEPT DRIFT:  P(Y | X) changes.     Detected by comparing PREDICTION ACCURACY against
                                          ACTUAL OUTCOMES once labels arrive
                                          (src/performance_monitor.py).

WHAT concept drift looks like here: imagine the coaching institute's exam board makes the
final exam noticeably harder next term (or changes the grading rubric) -- students with
EXACTLY the same study habits, attendance, and mock-test scores as before now end up with
LOWER final scores than the model (trained on last term's exam) would predict. The
STUDENTS haven't changed -- their study_hours, attendance_pct, city, income_bracket all
still look exactly like the training population. What changed is the RELATIONSHIP between
those inputs and the outcome.

HOW this module proves the point, concretely:
    1. `generate_concept_drift_batch()` draws students from the EXACT SAME feature
       distributions as data/raw/generate_data.py (same mean/std for study_hours,
       attendance_pct, etc.) -- so P(X) is, by construction, unchanged.
    2. It computes final_score with a DIFFERENT formula (studying/attendance now matter
       LESS -- the harder-exam scenario) -- so P(Y|X) has, by construction, changed.
    3. `run_concept_drift_check()` runs the existing model against this batch and shows,
       side by side:
           (a) src/drift.py's KS test on the INPUT features -> NOT significant (correctly:
               nothing about the inputs changed)
           (b) actual prediction error (MAE/RMSE/R2) against the NEW true labels -> BADLY
               DEGRADED (the model's learned relationship no longer matches reality)

THE KEY TEACHING POINT, stated explicitly (do not skip this when presenting the demo): a
KS-test-only monitoring setup would report "all clear" here. Only PERFORMANCE monitoring --
which requires actual outcome labels, and therefore is often DELAYED in production (see
src/performance_monitor.py's docstring) -- can catch this kind of drift. This is exactly
why a production monitoring system needs BOTH data drift checks (fast, label-free, always
available) AND performance monitoring (slower, needs delayed labels, but catches things
drift checks structurally cannot).

Run from the repo root:
    python -m src.concept_drift_demo
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = REPO_ROOT / "data" / "reference" / "training_reference.csv"
SCENARIO_PATH = REPO_ROOT / "data" / "scenarios" / "concept_drift_batch.csv"
REPORT_PATH = REPO_ROOT / "reports" / "concept_drift_report.json"

import sys
sys.path.insert(0, str(REPO_ROOT))
from src.predict import load_pipeline  # noqa: E402


def load_params() -> dict:
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def generate_concept_drift_batch(n: int = 300, seed: int = 999) -> pd.DataFrame:
    """Draws `n` students from the SAME feature distributions as data/raw/generate_data.py
    (P(X) unchanged) but labels them with a DIFFERENT study/attendance -> score relationship
    (P(Y|X) changed) -- simulating a harder exam / changed grading rubric next term.

    Returns a DataFrame with both the raw input columns (what the model sees) AND the true
    `final_score` under the NEW relationship (what performance monitoring needs, once these
    "arrive" as delayed labels).
    """
    rng = np.random.default_rng(seed)
    N = n
    SNAPSHOT_DATE = pd.Timestamp("2026-01-01")

    # ---- IDENTICAL feature-generation logic to data/raw/generate_data.py: this is what
    # makes P(X) provably unchanged, not just "probably similar". ------------------------
    study_hours = np.clip(rng.normal(10, 4, N), 0, 30)
    attendance_pct = np.clip(rng.normal(75, 15, N), 30, 100)
    ability = rng.normal(60, 15, N)
    mock_test_1 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
    mock_test_2 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
    mock_test_3 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
    income_bracket = rng.choice(["Low", "Medium", "High"], size=N, p=[0.35, 0.45, 0.20])
    city = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Pune"], size=N)
    enrollment_date = SNAPSHOT_DATE - pd.to_timedelta(rng.integers(30, 730, N), unit="D")
    shoe_size = rng.normal(8, 1.5, N)
    lucky_number = rng.integers(1, 100, N)

    # ---- CHANGED relationship (this is the concept drift): the harder-exam scenario.
    # Original:  final_score = 25 + 1.1*study_hours + 0.25*attendance_pct
    #                        + 0.015*study_hours*attendance_pct + 0.28*ability + income + noise
    # New:       studying and attendance now matter roughly HALF as much (the exam rewards
    #            raw aptitude -- `ability` -- more than preparation), and the baseline is
    #            lower (the exam itself got harder). Same inputs, same noise scale, visibly
    #            different outcome for the SAME student profile.
    income_effect = {"Low": -2.0, "Medium": 0.0, "High": 3.0}
    interaction_effect_new = 0.006 * study_hours * attendance_pct  # was 0.015
    final_score = (
        15  # was 25 -- harder exam, lower baseline
        + 0.5 * study_hours        # was 1.1
        + 0.12 * attendance_pct    # was 0.25
        + interaction_effect_new
        + 0.45 * ability           # was 0.28 -- raw aptitude now matters MORE
        + np.array([income_effect[b] for b in income_bracket])
        + rng.normal(0, 4, N)
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance_pct": attendance_pct.round(1),
        "mock_test_1": mock_test_1.round(1),
        "mock_test_2": mock_test_2.round(1),
        "mock_test_3": mock_test_3.round(1),
        "income_bracket": income_bracket,
        "city": city,
        "enrollment_date": enrollment_date.astype(str),
        "shoe_size": shoe_size.round(1),
        "lucky_number": lucky_number,
        "final_score": final_score,  # the NEW true label -- what the "delayed label" would be
    })
    return df


def run_concept_drift_check(n: int = 300, seed: int = 999) -> dict:
    """Runs the full side-by-side comparison and returns a report dict:
        - data_drift: the KS-test result on inputs (should show NO significant drift)
        - performance: model accuracy against the NEW true labels (should show CLEAR
          degradation vs. the model's own training-time baseline)
    """
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    from src.drift import run_ks_drift_check, NUMERIC_DRIFT_COLUMNS

    params = load_params()
    batch = generate_concept_drift_batch(n=n, seed=seed)

    SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(SCENARIO_PATH, index=False)

    # ---- (a) data drift check on the INPUTS only ---------------------------------------
    reference_df = pd.read_csv(REFERENCE_PATH)
    ks_report = run_ks_drift_check(
        reference_df, batch, NUMERIC_DRIFT_COLUMNS, params["drift"]["ks_p_value_threshold"]
    )
    n_drifted_features = int(ks_report["drift"].sum()) if not ks_report.empty else 0

    # ---- (b) actual model performance against the NEW true labels ----------------------
    pipeline = load_pipeline()
    feature_cols = [
        "study_hours", "attendance_pct", "mock_test_1", "mock_test_2", "mock_test_3",
        "income_bracket", "city", "enrollment_date", "shoe_size", "lucky_number",
    ]
    y_true = batch["final_score"]
    y_pred = pipeline.predict(batch[feature_cols])

    performance = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    baseline_path = REPO_ROOT / "reports" / "evaluation_report.json"
    baseline_metrics = None
    if baseline_path.exists():
        import json
        baseline_metrics = json.loads(baseline_path.read_text())["metrics"]

    result = {
        "scenario": "concept_drift (harder exam / changed grading rubric next term)",
        "n_samples": n,
        "data_drift": {
            "n_features_checked": len(ks_report),
            "n_features_drifted": n_drifted_features,
            "detail": ks_report.to_dict(orient="records"),
        },
        "performance_on_new_labels": performance,
        "baseline_performance": baseline_metrics,
    }
    REPORT_PATH.write_text(__import__("json").dumps(result, indent=2))
    return result


def print_report(result: dict) -> None:
    print(f"\n{'=' * 74}\nCONCEPT DRIFT DEMONSTRATION: {result['scenario']}\n{'=' * 74}")
    print(f"Simulated {result['n_samples']} students with UNCHANGED input distributions "
          f"but a CHANGED study/attendance -> score relationship.\n")

    dd = result["data_drift"]
    print(f"(a) Data drift check on INPUTS (KS test): "
          f"{dd['n_features_drifted']} / {dd['n_features_checked']} features flagged as drifted.")
    if dd["n_features_drifted"] == 0:
        print("    -> CORRECTLY shows no significant drift. P(X) really is unchanged.")
    else:
        print("    -> some features flagged (small-sample noise is possible here -- inspect "
              "reports/concept_drift_report.json).")

    perf = result["performance_on_new_labels"]
    print(f"\n(b) Model performance against the NEW true labels:")
    print(f"    MAE={perf['mae']:.2f}  RMSE={perf['rmse']:.2f}  R2={perf['r2']:.3f}")
    baseline = result.get("baseline_performance")
    if baseline:
        print(f"    (baseline, training-time test set: "
              f"MAE={baseline['mae']:.2f}  RMSE={baseline['rmse']:.2f}  R2={baseline['r2']:.3f})")
        rmse_increase = perf["rmse"] - baseline["rmse"]
        print(f"    -> RMSE increased by {rmse_increase:+.2f} versus baseline "
              f"({'DEGRADED -- concept drift' if rmse_increase > 0.5 else 'roughly unchanged'}).")

    print(f"\nTEACHING POINT: data drift = P(X) changes. Concept drift = P(Y|X) changes.")
    print(f"The KS test above (correctly) found nothing wrong with the inputs. Only "
          f"comparing predictions against ACTUAL labels (part (b)) reveals the model's "
          f"learned relationship no longer matches reality. A monitoring system that only "
          f"checks input drift would have missed this completely -- see "
          f"src/performance_monitor.py for how this project tracks performance decay in "
          f"a running system, where labels typically arrive LATE, not all at once like here.")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate concept drift vs. data drift.")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()
    result = run_concept_drift_check(n=args.n, seed=args.seed)
    print_report(result)


if __name__ == "__main__":
    main()
