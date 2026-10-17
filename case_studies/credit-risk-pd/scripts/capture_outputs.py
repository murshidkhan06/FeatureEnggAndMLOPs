"""Re-create docs/sample_outputs/*.txt (the REAL console output of every demo) and results.json.

    python scripts/capture_outputs.py            # runs everything in the right order, ~4 minutes
    python scripts/capture_outputs.py 04 12      # only the steps whose name starts with 04 or 12

Order matters (data -> features -> selection -> train -> ...).  The retraining demo ends with a ROLLBACK, so the champion is v1 again afterwards.
MLflow's pickle-security warnings are filtered out of the captured text (they are noise for a classroom).
A step command may be (command, regex): then only the lines matching the regex are kept (used to excerpt a long demo).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "sample_outputs"
OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1", MLFLOW_DISABLE_TELEMETRY="1", PYTHONIOENCODING="utf-8")
NOISE = re.compile(r"(WARNING mlflow|INFO mlflow|mlflow\.agent|PermutationExplainer|FutureWarning|UserWarning|^\s*warnings\.warn)")
PY = sys.executable
RD = "python -m src.monitoring.run_drift_check"
RT = "python -m src.models.retrain"


def run(cmd) -> str:
    keep = None
    if isinstance(cmd, tuple):
        cmd, keep = cmd
    args = [PY if c == "python" else c for c in shlex.split(cmd)]
    p = subprocess.run(args, cwd=ROOT, env=ENV, capture_output=True, text=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if not NOISE.search(l)]
    if keep:
        lines = [l for l in lines if re.search(keep, l)]
    text = "\n".join(lines)
    if p.returncode != 0:
        text += f"\n[exit code {p.returncode}]"
    return f"$ {cmd}\n{text}\n"


STEPS = [
    ("00_make_dataset", ["python -m src.data.make_dataset", "python -m src.features.build_features"]),
    ("01_data_profile", ["python -m src.data.profile_data"]),
    ("02_feature_manual_examples", ["python -m src.features.manual_examples"]),
    ("03_feature_selection", ["python -m src.features.select_features"]),
    ("04_leakage_demo", ["python -m src.features.leakage_demo"]),
    ("05_train_mlflow", ["python -m src.models.train --bootstrap-champion"]),
    ("06_evaluation", ["python -m src.models.evaluate"]),
    ("07_explainability", ["python -m src.models.explain"]),
    ("07b_explainability_shap_OPTIONAL", ["python -m src.models.explain --shap"]),
    ("08_pytest", ["python -m pytest -v -p no:cacheprovider"]),
    ("09_api_session", ["python scripts/api_session.py"]),
    ("10_drift_normal", [f"{RD} --prod data/production/batch_normal.csv"]),
    ("11_drift_shifted", [f"{RD} --prod data/production/batch_drifted.csv --today 2024-02-29",
                          f"{RD} --prod data/production/batch_drifted.csv --today 2024-05-15",
                          f"{RD} --prod data/production/batch_drifted.csv --today 2025-06-30"]),
    ("11b_drift_data_quality_issue", [f"{RD} --prod data/production/batch_dq_issue.csv"]),
    ("11c_label_lag_timeline", ["python -m src.monitoring.perf_monitor --today 2024-02-29", "python -m src.monitoring.perf_monitor --today 2025-06-30"]),
    ("12_retrain_promotion", [f"{RT} --prod data/production/batch_normal.csv",
                              f"{RT} --prod data/production/batch_dq_issue.csv",
                              f"{RT} --prod data/production/batch_drifted.csv --today 2024-02-29",
                              f"{RT} --prod data/production/batch_drifted.csv --min-improvement 0.05 --approve-as 'Priya Nair (Model Risk)'",
                              f"{RT} --prod data/production/batch_drifted.csv",
                              f"{RT} --prod data/production/batch_drifted.csv --approve-as 'Priya Nair (Model Risk)'",
                              f"{RT} --rollback"]),
    ("13_failure_scenarios", [("python -m src.features.leakage_demo", r"(ACT|single-feature|average_balance|RIGHT|WRONG|GO-LIVE|live AUC|model  |dpd_target|recoveries|collections_status|feature  )"),
                              f"{RD} --prod data/production/batch_dq_issue.csv --today 2025-06-30",
                              f"{RT} --prod data/production/batch_drifted.csv --window stress_only --calm-check"]),
    ("14_batch_score", ["python -m src.models.batch_score"]),
    ("15_feature_registry", ["python -m src.governance.feature_lifecycle table", "python -m src.governance.feature_lifecycle history",
                             "python -m src.governance.feature_lifecycle validate", "python -m src.governance.feature_lifecycle gate-demo"]),
    ("16_quality_gate", ["python scripts/quality_gate.py"]),
    ("17_traffic_drift_exporter", ["python scripts/traffic_demo.py"]),
    ("18_static_checks", ["python scripts/static_checks.py"]),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, cmds in STEPS:
        if only and not any(name.startswith(o) for o in only):
            continue
        print("running", name, flush=True)
        (OUT / f"{name}.txt").write_text("\n".join(run(c) for c in cmds), encoding="utf-8")
    if not only:
        print(run("python scripts/build_results.py"))
