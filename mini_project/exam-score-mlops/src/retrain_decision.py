"""
src/retrain_decision.py

The TRIGGER-BASED RETRAINING decision: the piece that sits BETWEEN monitoring/drift/
alerting (which only ever OBSERVE and REPORT) and src/retrain.py (which only ever TRAINS a
candidate once told to). This module answers one question, explicitly and auditably:

    "Given everything we currently know, should we retrain right now?"

THE CENTRAL TEACHING POINT OF THIS MODULE (do not simplify this away): drift detected does
NOT, by itself, mean retrain. A production ML system that retrains automatically the moment
src/drift.py flags ANY feature is a system that will retrain constantly on noise -- small
production samples routinely trip a KS test (see src/drift.py's own troubleshooting note),
and even a REAL, confirmed distribution shift does not necessarily mean the model's actual
accuracy has degraded (see src/concept_drift_demo.py for a case where accuracy tanks with
ZERO data drift, and plenty of real cases go the other way -- inputs shift, accuracy barely
moves).

THE DECISION TABLE this module implements (`params.yaml: retraining.*` controls the
thresholds involved):

    | Data drift | Performance decay | New data volume  | Decision      |
    |------------|--------------------|-----------------|---------------|
    | No         | No                 | (any)           | NO_RETRAIN    |
    | Yes        | No                 | (any)           | NO_RETRAIN*   |
    | Yes/No     | Yes                | >= min required | RETRAIN       |
    | Yes/No     | Yes                | < min required  | NO_RETRAIN**  |

    *  Drift alone is evidence to WATCH more closely, not evidence to act on -- this is
       exactly Scenario 2 from the course spec ("data drift -> alert -> performance still
       acceptable -> no retraining").
    ** Confirmed performance decay is serious, but retraining on too little new data can
       itself produce a worse, noisier model -- this module still recommends waiting for
       more labeled data rather than retraining on a handful of rows (this is the "training
       cost / deployment risk" side of the cost-vs-performance trade-off the course spec
       asks for, made concrete).

WHY performance decay is the deciding factor and drift is only supporting evidence: drift
tells you the WORLD changed; performance decay tells you the MODEL got WORSE because of it.
Only the second one is actually costing the business anything. (Set
`params.yaml: retraining.require_performance_decay_to_trigger: false` to instead trigger
retraining on data drift alone -- deliberately left available as a config knob so an
instructor can demonstrate BOTH policies, but `true` -- the more conservative, defensible
default -- is what this project ships with.)

This module does NOT itself decide whether a retrained candidate is actually GOOD -- that
remains src/retrain.py's job (the champion/candidate promotion gate), kept as a completely
separate, later decision. "Should we retrain?" and "was the retrain any good?" are two
different questions on purpose (see Scenarios 4 and 5 in the classroom demo).

Run from the repo root:
    python -m src.retrain_decision
    python -m src.retrain_decision --execute     # also runs src.retrain if RETRAIN is decided
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "reports" / "retrain_decision_report.json"

import sys
sys.path.insert(0, str(REPO_ROOT))
from src.drift import check_drift, check_categorical_drift  # noqa: E402
from src.performance_monitor import check_performance_decay, load_labeled_predictions  # noqa: E402


def load_params() -> dict:
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def decide_retraining(params: dict | None = None) -> Dict[str, Any]:
    params = params or load_params()
    retrain_cfg = params["retraining"]

    # ---- 1. data drift (informational input, not the deciding factor) ------------------
    try:
        numeric_report = check_drift(p_value_threshold=params["drift"]["ks_p_value_threshold"])
        numeric_drifted = list(numeric_report[numeric_report["drift"]]["feature"]) if not numeric_report.empty else []
    except FileNotFoundError:
        numeric_drifted = []
    try:
        categorical_report = check_categorical_drift(p_value_threshold=params["drift"]["categorical_p_value_threshold"])
        categorical_drifted = list(categorical_report[categorical_report["drift"]]["feature"]) if not categorical_report.empty else []
    except FileNotFoundError:
        categorical_drifted = []
    data_drift_detected = bool(numeric_drifted or categorical_drifted)

    # ---- 2. performance decay (the deciding factor) -------------------------------------
    performance_result = check_performance_decay(params)
    performance_decayed = performance_result["decayed"]

    # ---- 3. new-data volume gate --------------------------------------------------------
    joined = load_labeled_predictions()
    n_new_rows = len(joined)
    min_required = retrain_cfg["min_new_rows_to_consider_retrain"]
    enough_volume = n_new_rows >= min_required

    # ---- decision table -------------------------------------------------------------------
    require_decay = retrain_cfg["require_performance_decay_to_trigger"]
    if require_decay:
        deciding_condition_met = performance_decayed
    else:
        deciding_condition_met = performance_decayed or data_drift_detected

    if deciding_condition_met and enough_volume:
        decision = "RETRAIN"
        reason = (
            f"{'Performance decay' if performance_decayed else 'Data drift'} confirmed "
            f"with {n_new_rows} new labeled rows (>= {min_required} required)."
        )
    elif deciding_condition_met and not enough_volume:
        decision = "NO_RETRAIN"
        reason = (
            f"{'Performance decay' if performance_decayed else 'Data drift'} detected, but only "
            f"{n_new_rows} new labeled rows available (< {min_required} required) -- waiting "
            f"for more data before retraining on a small, noisy batch."
        )
    elif data_drift_detected and not performance_decayed:
        decision = "NO_RETRAIN"
        reason = (
            "Data drift detected, but performance has NOT meaningfully decayed -- input "
            "distribution changed without hurting real accuracy (yet). Keep monitoring; "
            "this is not, by itself, a reason to retrain."
        )
    else:
        decision = "NO_RETRAIN"
        reason = "No data drift and no performance decay detected -- system operating normally."

    result = {
        "decision": decision,
        "reason": reason,
        "evidence": {
            "data_drift_detected": data_drift_detected,
            "numeric_features_drifted": numeric_drifted,
            "categorical_features_drifted": categorical_drifted,
            "performance_decayed": performance_decayed,
            "performance_status": performance_result["status"],
            "n_new_labeled_rows": n_new_rows,
            "min_new_rows_required": min_required,
        },
        "policy": {
            "require_performance_decay_to_trigger": require_decay,
        },
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    return result


def print_decision(result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 66}\nRetraining decision: {result['decision']}\n{'=' * 66}")
    print(f"Reason: {result['reason']}\n")
    ev = result["evidence"]
    print(f"Data drift detected:       {ev['data_drift_detected']} "
          f"(numeric: {ev['numeric_features_drifted'] or 'none'}; "
          f"categorical: {ev['categorical_features_drifted'] or 'none'})")
    print(f"Performance decayed:       {ev['performance_decayed']} (status={ev['performance_status']})")
    print(f"New labeled rows:          {ev['n_new_labeled_rows']} (need >= {ev['min_new_rows_required']})")


def main():
    parser = argparse.ArgumentParser(description="Decide whether a retraining trigger should fire.")
    parser.add_argument("--execute", action="store_true",
                         help="If the decision is RETRAIN, also run src.retrain immediately "
                              "(this is the 'trigger -> Kubeflow pipeline -> train' wiring; "
                              "kept as an explicit opt-in flag rather than automatic, per the "
                              "course spec's 'a human decision, not an unattended retrain-and-"
                              "deploy' guidance).")
    args = parser.parse_args()

    result = decide_retraining()
    print_decision(result)

    if result["decision"] == "RETRAIN" and args.execute:
        print("\n--execute set: running src.retrain now...")
        from src.retrain import retrain
        retrain()


if __name__ == "__main__":
    main()
