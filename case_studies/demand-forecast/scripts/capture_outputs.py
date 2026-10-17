"""Re-create docs/sample_outputs/*.txt (the REAL console output of every demo) and results.json.

    python scripts/capture_outputs.py            # runs everything in the right order, ~5 minutes

Order matters (data -> features -> train -> ...).  The retraining demo ends with a ROLLBACK, so the champion is v1 again afterwards.
MLflow's pickle-security warnings are filtered out of the captured text (they are noise for a classroom).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "sample_outputs"
OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1", MLFLOW_DISABLE_TELEMETRY="1", PYTHONIOENCODING="utf-8")
NOISE = re.compile(r"(WARNING mlflow|INFO mlflow|mlflow\.agent|PermutationExplainer|FutureWarning|UserWarning|^\s*warnings\.warn)")
PY = sys.executable
RD = "python -m src.monitoring.run_drift_check"


def run(cmd: str) -> str:
    args = [PY if c == "python" else c for c in cmd.split()]
    p = subprocess.run(args, cwd=ROOT, env=ENV, capture_output=True, text=True)
    text = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not NOISE.search(l))
    if p.returncode != 0:
        text += f"\n[exit code {p.returncode}]"
    return f"$ {cmd}\n{text}\n"


STEPS = [
    ("00_make_dataset", ["python -m src.data.make_dataset"]),
    ("01_data_profile", ["python -m src.data.profile_data"]),
    ("02_feature_manual_examples", ["python -m src.features.manual_examples"]),
    ("03_feature_selection", ["python -m src.features.select_features"]),
    ("04_leakage_demo", ["python -m src.features.leakage_demo"]),
    ("05_train_mlflow", ["python -m src.models.train --bootstrap-champion"]),
    ("06_evaluation", ["python -m src.models.evaluate"]),
    ("07_explainability", ["python -m src.models.explain"]),
    ("08_pytest", ["python -m pytest -v -p no:cacheprovider"]),
    ("09_api_session", ["python scripts/api_session.py"]),
    ("10_drift_normal", [f"{RD} --prod data/production/batch_normal.csv"]),
    ("10b_drift_normal_seasonal_reference", [f"{RD} --prod data/production/batch_normal.csv --ref-window seasonal"]),
    ("11_drift_shifted", [f"{RD} --prod data/production/batch_drifted.csv"]),
    ("11b_drift_festival_expected_shift", [f"{RD} --prod data/production/batch_festival.csv"]),
    ("11c_drift_festival_seasonal_reference", [f"{RD} --prod data/production/batch_festival.csv --ref-window seasonal"]),
    ("11d_performance_monitor", ["python -m src.monitoring.performance --prod data/production/batch_normal.csv",
                                 "python -m src.monitoring.performance --prod data/production/batch_drifted.csv"]),
    ("12_retrain_promotion", ["python -m src.models.retrain --prod data/production/batch_normal.csv",
                              "python -m src.models.retrain --prod data/production/batch_normal.csv --mode scheduled",
                              "python -m src.models.retrain --prod data/production/batch_drifted.csv --compare-strategies",
                              "python -m src.models.retrain --prod data/production/batch_drifted.csv --force-model hgb --approve-as planner_asha",
                              "python -m src.models.retrain --prod data/production/batch_drifted.csv --approve-as planner_asha --demo-rollback"]),
    ("13_failure_scenarios", ["python -m src.monitoring.failure_scenarios"]),
    ("14_trend_shift_extrapolation", ["python -m src.models.trend_shift"]),
    ("15_batch_forecast", ["python -m src.models.batch_forecast", "python -m src.models.batch_forecast --backtest"]),
    ("16_weekly_job", ["python -m src.models.weekly_job --new-actuals data/production/batch_drifted.csv --through 2025-01-07"]),
    ("17_quality_gate", ["python scripts/quality_gate.py"]),
    ("18_rolling_origin_optional", ["python -m src.models.rolling_origin"]),
    ("19_traffic_drift_exporter", ["python scripts/traffic_demo.py"]),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, cmds in STEPS:
        if only and not any(name.startswith(o) for o in only):
            continue
        print("running", name, flush=True)
        (OUT / f"{name}.txt").write_text("\n".join(run(c) for c in cmds), encoding="utf-8")
    png = ROOT / "reports" / "trend_shift_extrapolation.png"
    if png.exists():
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "trend_shift_extrapolation.png").write_bytes(png.read_bytes())     # figure used in the README
    if not only:
        print(run("python scripts/build_results.py"))
