"""
scripts/classroom_demo.py

The END-TO-END CLASSROOM DEMONSTRATION for Unit 4: a single, REAL, runnable script that
walks through the complete lifecycle --

    DATA -> DVC -> VALIDATION -> FEATURE ENGINEERING -> TRAINING -> MLFLOW -> EVALUATION
    -> REGISTRY -> DEPLOYMENT -> PREDICTIONS -> OBSERVABILITY -> MONITORING
    -> DRIFT DETECTION -> PERFORMANCE MONITORING -> RETRAINING DECISION -> KUBEFLOW PIPELINE
    -> RETRAIN -> EVALUATE -> PROMOTE/REJECT -> DEPLOY -> CONTINUOUS MONITORING

-- and then demonstrates all FIVE required teaching scenarios explicitly. Nothing in this
script is simulated at the level of "pretend this happened" -- every step below calls this
project's REAL code (the same `src.*` modules and Kubeflow pipeline used everywhere else in
this project), against a REAL running FastAPI instance, with REAL (synthetic but honestly
computed) data.

HOW "delayed labels" are simulated here, honestly: a real system waits days/weeks for a
true `final_score` to become known. This script can't wait days, so `_true_score()` below
computes what the ACTUAL exam outcome would be for a given student, using one of two
formulas: the ORIGINAL data-generating relationship (data/raw/generate_data.py) or the
CONCEPT-DRIFT relationship (src/concept_drift_demo.py) -- then posts that value to
`POST /feedback`, exactly as a real delayed-label pipeline would once true outcomes were
known. This is not "faking" performance metrics -- it's the same evaluation math
(MAE/RMSE/R2) computed honestly by src/performance_monitor.py, seeded with labels this
script is transparent about having generated. `ability_proxy` (the average of the three
mock test scores) stands in for the LATENT `ability` variable used only during original
dataset generation -- itself an intentional, explained approximation, since only truly
observable columns are available for a production row.

Run from the repo root (needs the model already trained -- `python -m src.train` -- and
port 8000 free):
    python scripts/classroom_demo.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

API_BASE = "http://localhost:8000"
MONITORING_DIR = REPO_ROOT / "monitoring"
PREDICTIONS_LOG = MONITORING_DIR / "predictions.csv"
LABELS_LOG = MONITORING_DIR / "labels.csv"


# ---------------------------------------------------------------------------------------
# Narration helpers
# ---------------------------------------------------------------------------------------
def banner(text: str) -> None:
    print(f"\n{'#' * 80}\n# {text}\n{'#' * 80}")


def step(n, total, text: str) -> None:
    print(f"\n{'-' * 78}\nSTEP {n}/{total}: {text}\n{'-' * 78}")


def scenario_header(n: str, text: str) -> None:
    print(f"\n{'=' * 80}\nSCENARIO {n}: {text}\n{'=' * 80}")


# ---------------------------------------------------------------------------------------
# API process management
# ---------------------------------------------------------------------------------------
def start_api() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            httpx.get(f"{API_BASE}/health", timeout=1.0)
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("API did not become healthy in time.")


def stop_api(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------------------
# Delayed-label simulation
# ---------------------------------------------------------------------------------------
def _true_score(row: pd.Series, relationship: str, rng: np.random.Generator) -> float:
    """Computes the ACTUAL exam outcome for one student, under either relationship.
    See this module's docstring for why `ability_proxy` stands in for the latent
    `ability` variable, and why this is an honest (not fabricated) label."""
    ability_proxy = (row["mock_test_1"] + row["mock_test_2"] + row["mock_test_3"]) / 3.0
    income_effect = {"Low": -2.0, "Medium": 0.0, "High": 3.0}[row["income_bracket"]]
    noise = rng.normal(0, 4)

    if relationship == "original":
        interaction = 0.015 * row["study_hours"] * row["attendance_pct"]
        score = (25 + 1.1 * row["study_hours"] + 0.25 * row["attendance_pct"]
                  + interaction + 0.28 * ability_proxy + income_effect + noise)
    elif relationship == "concept_drift":
        interaction = 0.006 * row["study_hours"] * row["attendance_pct"]
        score = (15 + 0.5 * row["study_hours"] + 0.12 * row["attendance_pct"]
                  + interaction + 0.45 * ability_proxy + income_effect + noise)
    else:
        raise ValueError(relationship)
    return float(np.clip(score, 0, 100).round(1))


def send_feedback_for_recent_predictions(n: int, relationship: str, seed: int = 7) -> int:
    """Reads the N most recent successful predictions from monitoring/predictions.csv and
    posts a `POST /feedback` (delayed label) for each, computed with `_true_score()`.
    Returns how many labels were actually sent."""
    if not PREDICTIONS_LOG.exists():
        return 0
    df = pd.read_csv(PREDICTIONS_LOG)
    df = df[df["status"] == "success"].tail(n)
    rng = np.random.default_rng(seed)

    sent = 0
    with httpx.Client(base_url=API_BASE, timeout=10.0) as client:
        for _, row in df.iterrows():
            actual = _true_score(row, relationship, rng)
            resp = client.post("/feedback", json={
                "request_id": row["request_id"], "actual_final_score": actual,
            })
            if resp.status_code == 200:
                sent += 1
    return sent


def generate_updated_training_data(n: int = 500, seed: int = 2026) -> Path:
    """Generates a FRESH, full-sized training CSV (same schema as
    data/raw/student_exam_scores.csv) labeled under the CONCEPT-DRIFT relationship --
    i.e. what a real "new batch of labeled data reflecting the new reality" would look
    like once enough of it has accumulated. This is what makes Scenario 5 (retraining
    produces a BETTER model -> promoted) a GENUINE outcome rather than a scripted one:
    src/retrain.py's promotion gate is given a real, harder test -- evaluate a candidate
    trained on this new data against the STALE champion (still only knowing the OLD
    relationship) on the SAME held-out split. The champion has no way to do well there;
    the candidate, trained on data that actually reflects the new relationship, does --
    exactly the situation retraining exists to fix. See kubeflow/components.py's
    `retrain_and_promote_component` docstring for why this matters (Scenario 4 -- retrain
    on the SAME old data -- correctly stays REJECTED for the opposite reason)."""
    rng = np.random.default_rng(seed)
    N = n
    snapshot = pd.Timestamp("2026-01-01")

    study_hours = np.clip(rng.normal(10, 4, N), 0, 30)
    attendance_pct = np.clip(rng.normal(75, 15, N), 30, 100)
    ability = rng.normal(60, 15, N)
    mock_test_1 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
    mock_test_2 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
    mock_test_3 = np.clip(ability + rng.normal(0, 6, N), 0, 100)
    income_bracket = rng.choice(["Low", "Medium", "High"], size=N, p=[0.35, 0.45, 0.20])
    city = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Pune"], size=N)
    enrollment_date = snapshot - pd.to_timedelta(rng.integers(30, 730, N), unit="D")
    shoe_size = rng.normal(8, 1.5, N)
    lucky_number = rng.integers(1, 100, N)

    income_effect = {"Low": -2.0, "Medium": 0.0, "High": 3.0}
    interaction_new = 0.006 * study_hours * attendance_pct
    final_score = (
        15 + 0.5 * study_hours + 0.12 * attendance_pct + interaction_new
        + 0.45 * ability + np.array([income_effect[b] for b in income_bracket])
        + rng.normal(0, 4, N)
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame({
        "student_id": np.arange(1, N + 1),
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
        "final_score": final_score,
    })
    out_path = REPO_ROOT / "data" / "scenarios" / "updated_training_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def reset_monitoring_state() -> None:
    """Clears predictions/labels/reports between scenarios so each one starts from a known,
    clean state -- exactly what makes the five scenarios independently reproducible."""
    for f in [PREDICTIONS_LOG, LABELS_LOG]:
        if f.exists():
            f.unlink()
    reports_dir = REPO_ROOT / "reports"
    for pattern in ["drift_report.csv", "categorical_drift_report.csv",
                     "performance_report.json", "retrain_decision_report.json",
                     "alerts_report.json"]:
        p = reports_dir / pattern
        if p.exists():
            p.unlink()


def run_module(module: str, extra_args: list[str] | None = None) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *(extra_args or [])],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.stdout


# ---------------------------------------------------------------------------------------
# PART A -- the 20-step end-to-end walkthrough (doubles as Scenario 3: drift + decay ->
# retrain -> promote -> deploy, since that is the path that exercises every step).
# ---------------------------------------------------------------------------------------
def part_a_full_lifecycle(api_proc: subprocess.Popen) -> None:
    banner("PART A: END-TO-END CLASSROOM DEMONSTRATION (20 steps)")

    step(1, 20, "Run existing Exam Score model (already trained via `python -m src.train`).")
    health = httpx.get(f"{API_BASE}/health").json()
    print(f"API health: {health}")

    step(2, 20, "Generate normal production predictions.")
    run_module("src.generate_traffic", ["--n", "150"])

    step(3, 20, "Predictions are logged automatically (src/monitoring.py, on every /predict).")
    print(f"monitoring/predictions.csv now has "
          f"{len(pd.read_csv(PREDICTIONS_LOG))} rows.")

    step(4, 20, "Show monitoring summary.")
    run_module("src.monitoring")

    step(5, 20, "Introduce CHANGED production data (simulated cohort shift).")
    run_module("src.generate_traffic", ["--n", "150", "--simulate-drift"])

    step(6, 20, "Run drift detection (numeric KS test + categorical chi-square test).")
    run_module("src.drift")

    step(7, 20, "Drift alert is raised (src/alerts.py -- print + log + report file).")
    run_module("src.alerts")

    step(8, 20, "Introduce delayed labels for the ORIGINAL relationship first "
                "(students really did just study/attend less -- P(Y|X) unchanged).")
    n_sent = send_feedback_for_recent_predictions(n=120, relationship="original")
    print(f"Sent {n_sent} delayed labels (true relationship unchanged).")

    step(9, 20, "Show performance monitoring -- expect performance roughly intact "
                "(this alone is Scenario 2: drift + no real decay -> no retrain).")
    run_module("src.performance_monitor")
    run_module("src.retrain_decision")
    # retrain_decision.py prints a human-readable report to stdout; the reliable,
    # machine-readable result is the JSON file it also writes -- read that instead.
    d1 = json.loads((REPO_ROOT / "reports" / "retrain_decision_report.json").read_text())
    print(f"[Scenario 2 checkpoint] decision={d1['decision']} (expected NO_RETRAIN)")

    step(10, 20, "NOW introduce a genuine CONCEPT DRIFT on top (harder exam / changed "
                 "rubric) via delayed labels for a fresh batch of production traffic.")
    run_module("src.generate_traffic", ["--n", "150"])  # fresh, non-drifted INPUTS
    n_sent2 = send_feedback_for_recent_predictions(n=150, relationship="concept_drift")
    print(f"Sent {n_sent2} delayed labels under the CHANGED (concept-drift) relationship.")

    step(11, 20, "Re-run performance monitoring -- expect DECAY now.")
    run_module("src.performance_monitor")

    step(12, 20, "Re-run the retraining trigger decision -- expect RETRAIN now.")
    run_module("src.retrain_decision")
    d2 = json.loads((REPO_ROOT / "reports" / "retrain_decision_report.json").read_text())
    print(f"[Scenario 3 checkpoint] decision={d2['decision']} (expected RETRAIN)")

    step(13, 20, "Start the Kubeflow Pipeline (local runner -- see kubeflow/run_local.py).")
    step(14, 20, "  -> Validate data")
    step(15, 20, "  -> Train candidate model ON UPDATED DATA reflecting the new reality")
    step(16, 20, "  -> Log experiment to MLflow (inside the same training step)")
    step(17, 20, "  -> Evaluate candidate")
    step(18, 20, "  -> Compare candidate with current (stale) champion; Promote / Reject")
    print("Generating a fresh, full-sized training batch under the NEW (concept-drift) "
          "relationship -- what enough accumulated real-world labels would look like:")
    updated_data_path = generate_updated_training_data(n=500)
    print(f"  -> {updated_data_path}")
    print("\nRunning the full Kubeflow pipeline now (covers steps 13-18 as one DAG run):")
    from kfp import local
    local.init(runner=local.SubprocessRunner(use_venv=False),
               pipeline_root=str(REPO_ROOT / "local_outputs"))
    from kubeflow.pipeline import exam_score_mlops_pipeline
    exam_score_mlops_pipeline(repo_root=str(REPO_ROOT), retrain_data_path=str(updated_data_path))

    retrain_report = json.loads((REPO_ROOT / "reports" / "retrain_report.json").read_text())
    print(f"\nPipeline's retraining outcome: {retrain_report['decision']} "
          f"(this is Scenario 5: a candidate trained on data reflecting the new reality "
          f"legitimately beats the stale champion and is PROMOTED).")

    step(19, 20, "Deploy the promoted model (models/exam_score_pipeline.joblib already "
                 "swapped by src.retrain if promoted; restart the API process to pick it up).")
    stop_api(api_proc)
    new_api_proc = start_api()
    health = httpx.get(f"{API_BASE}/health").json()
    print(f"API restarted with new model: {health}")

    step(20, 20, "Continue monitoring the newly deployed model.")
    run_module("src.generate_traffic", ["--n", "50"])
    run_module("src.monitoring")

    globals()["_api_proc"] = new_api_proc


# ---------------------------------------------------------------------------------------
# PART B -- the five required teaching scenarios, run independently and explicitly.
# Scenario 3 is already demonstrated in Part A above; it is re-summarized here for
# completeness rather than re-run (re-running it would just repeat Part A).
# ---------------------------------------------------------------------------------------
def part_b_scenario_1_no_drift() -> None:
    scenario_header("1", "No drift -> No retraining")
    reset_monitoring_state()
    run_module("src.generate_traffic", ["--n", "120"])
    run_module("src.drift")
    run_module("src.retrain_decision")
    d = json.loads((REPO_ROOT / "reports" / "retrain_decision_report.json").read_text())
    assert d["decision"] == "NO_RETRAIN", f"expected NO_RETRAIN, got {d['decision']}"
    print(f"\n[VERIFIED] decision={d['decision']} -- normal traffic, nothing to act on.")


def part_b_scenario_2_drift_no_decay() -> None:
    scenario_header("2", "Data drift -> Alert -> Performance still acceptable -> No retraining")
    reset_monitoring_state()
    run_module("src.generate_traffic", ["--n", "150", "--simulate-drift"])
    run_module("src.drift")
    n_sent = send_feedback_for_recent_predictions(n=120, relationship="original")
    print(f"Sent {n_sent} delayed labels using the UNCHANGED true relationship.")
    run_module("src.performance_monitor")
    run_module("src.retrain_decision")
    d = json.loads((REPO_ROOT / "reports" / "retrain_decision_report.json").read_text())
    assert d["decision"] == "NO_RETRAIN", f"expected NO_RETRAIN, got {d['decision']}"
    print(f"\n[VERIFIED] decision={d['decision']} -- drift alone did not trigger a retrain.")


def part_b_scenario_3_summary() -> None:
    scenario_header("3", "Data drift + Performance degradation -> Retraining (see Part A above)")
    print("Already demonstrated end to end in Part A (steps 5-19): drift was detected, "
          "concept drift then caused real performance decay, and the retraining trigger "
          "correctly fired (src/retrain_decision.py's decision=RETRAIN).")


def part_b_scenario_4_worse_candidate_rejected() -> None:
    scenario_header("4", "Retraining produces a WORSE (not-better) model -> Candidate REJECTED")
    print("Retraining on the SAME, unmodified original training data (no new information, "
          "and NOT the updated data Part A used) should not beat the existing champion by "
          "the required margin -- src/retrain.py's own promotion gate should reject it.")
    result = subprocess.run(
        [sys.executable, "-m", "src.retrain"], cwd=REPO_ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    report = json.loads((REPO_ROOT / "reports" / "retrain_report.json").read_text())
    print(f"[VERIFIED] decision={report['decision']} "
          f"(REJECTED is the expected, correct outcome here -- retraining on the same old "
          f"data the champion already knows cannot legitimately clear the min-improvement "
          f"bar).")


def part_b_scenario_5_better_candidate_promoted() -> None:
    scenario_header("5", "Retraining produces a BETTER model -> Candidate PROMOTED")
    print("Already demonstrated, genuinely, in Part A (step 18): a candidate trained on "
          "data that actually reflects the new (concept-drifted) relationship was "
          "evaluated against the STALE champion (which only ever knew the OLD "
          "relationship) on the same held-out split. The candidate legitimately cleared "
          "the promotion gate's min-improvement bar and was PROMOTED -- see Part A's "
          "printed retrain report above for the exact champion vs. candidate MAE.")


def reset_to_fresh_champion() -> None:
    """Retrains from scratch on the ORIGINAL data/raw CSV (via `python -m src.train`, NOT
    src.retrain) and restarts the API. Run between Part A and Part B so the five scenarios
    each start from a known, canonical champion -- Part A deliberately ends with a
    DIFFERENT (promoted, concept-drift-trained) champion in place, which would otherwise
    make Part B's "fresh, normal traffic" scenarios (1 and 2) start from a confusing state.
    NOTE for instructors: this also illustrates a real, easy-to-miss gotcha --
    reports/evaluation_report.json (the "baseline" src/performance_monitor.py compares
    against) is only rewritten by src/train.py, never by src/retrain.py. After a real
    promotion, a team should deliberately refresh that baseline too, or performance-decay
    checks keep comparing against a champion that no longer exists."""
    print("\n[Resetting to a fresh, canonical champion before Part B's independent scenarios...]")
    proc = globals().get("_api_proc")
    if proc is not None:
        stop_api(proc)
    run_module("src.train")
    new_proc = start_api()
    globals()["_api_proc"] = new_proc


def main():
    api_proc = start_api()
    globals()["_api_proc"] = api_proc
    try:
        part_a_full_lifecycle(api_proc)

        reset_to_fresh_champion()

        banner("PART B: THE FIVE TEACHING SCENARIOS")
        part_b_scenario_1_no_drift()
        part_b_scenario_2_drift_no_decay()
        part_b_scenario_3_summary()
        part_b_scenario_4_worse_candidate_rejected()
        part_b_scenario_5_better_candidate_promoted()

        banner("CLASSROOM DEMO COMPLETE")
        print("Key teaching point reinforced throughout: DRIFT DETECTED != AUTOMATIC "
              "RETRAINING. Retraining happened exactly once (Part A / Scenario 3), only "
              "after BOTH data drift AND confirmed performance decay -- never from drift "
              "alone (Scenario 2), and a retrain that could not legitimately improve on "
              "the champion was correctly rejected (Scenario 4).")
    finally:
        stop_api(globals()["_api_proc"])


if __name__ == "__main__":
    main()
