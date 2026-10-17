"""Run every demo of the project in the right order and save the console output of each one in docs/sample_outputs/.

WHY: (1) the saved files are the "expected output" students compare with; (2) running this script is the quickest way to
check that the WHOLE project works on your machine (it stops with an error message if any step fails).

    python scripts/capture_outputs.py              # full run (about 6-8 minutes; the leakage demo is the slowest step)
    python scripts/capture_outputs.py --skip-api   # skip the steps that start the web API

Order matters: data -> features -> selection -> train -> evaluate/explain -> governance -> tests -> API + monitoring -> retraining.
The retraining step comes LAST because it replaces (then, in the demo, rolls back) the champion model.
MLflow prints harmless INFO / pickle warnings on stderr; those lines are removed from the saved files.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import os
import re
import socket
import subprocess
import time

import requests

OUT = ROOT / "docs" / "sample_outputs"
NOISE = re.compile(r"^\d{4}/\d\d/\d\d \d\d:\d\d:\d\d (INFO|WARNING) mlflow|UserWarning|warnings\.warn")
ENV = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "MLFLOW_DISABLE_TELEMETRY": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
PY = sys.executable


def run(name: str, *commands: list[str], must_succeed: bool = True) -> None:
    """Run one or more commands, write '$ command' + output to docs/sample_outputs/<name>.txt"""
    OUT.mkdir(parents=True, exist_ok=True)
    chunks, t0 = [], time.time()
    for cmd in commands:
        shown = " ".join(["python" if c == PY else c for c in cmd])
        p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
        text = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not NOISE.search(l))
        text = text.replace(str(ROOT), "<project root>")          # no machine-specific paths in the saved files
        chunks.append(f"$ {shown}\n{text.rstrip()}\n" + ("" if p.returncode == 0 else f"[exit code {p.returncode}]\n"))
        if p.returncode != 0 and must_succeed:
            (OUT / f"{name}.txt").write_text("\n".join(chunks), encoding="utf-8")
            sys.exit(f"STEP {name} FAILED (exit {p.returncode}):\n{text[-1500:]}")
    (OUT / f"{name}.txt").write_text("\n".join(chunks), encoding="utf-8")
    print(f"  ok  {name:34s} ({time.time() - t0:5.1f}s)")


def m(module: str, *args: str) -> list[str]:
    return [PY, "-m", module, *args]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-api", action="store_true")
    a = ap.parse_args()
    print(f"capturing demo outputs into {OUT}")
    run("00_make_data_and_split", m("src.data.make_dataset"), m("src.features.build_features"))
    run("01_data_profile", m("src.data.profile_data"))
    run("02_feature_manual_examples", m("src.features.manual_examples"))
    run("03_feature_selection", m("src.features.select_features"))
    run("05_train_mlflow", m("src.models.train", "--bootstrap-champion"))
    run("06_evaluation", m("src.models.evaluate"))
    run("07_explainability", m("src.models.explain"))
    run("04_leakage_demo", m("src.features.leakage_demo"))
    run("13_fairness_audit", m("src.governance.fairness_audit"))
    run("14_age_experiment", m("src.governance.age_experiment"))
    run("15_review_pack_and_gate", m("src.governance.model_card"), m("src.governance.review_gate"),
        m("src.governance.review_gate", "--demo-bad-model"), must_succeed=False)
    run("08_pytest", [PY, "-m", "pytest", "-v", "-p", "no:cacheprovider"], [PY, "scripts/quality_gate.py"])
    run("16_failure_scenarios", m("src.monitoring.failure_scenarios"))

    if not a.skip_api:
        (ROOT / "logs").mkdir(exist_ok=True)
        log = ROOT / "logs" / "predictions.jsonl"
        if log.exists():
            log.unlink()
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        server = subprocess.Popen([PY, "-m", "uvicorn", "src.api.app:app", "--port", str(port)], cwd=ROOT, env=ENV,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(60):
                try:
                    if requests.get(f"{url}/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    time.sleep(0.5)
            else:
                sys.exit("API did not start")
            run("09b_traffic_and_drift_exporter",
                [PY, "scripts/smoke_test.py", "--url", url],
                [PY, "scripts/send_traffic.py", "--csv", "data/production/batch_normal.csv", "--n", "300", "--url", url],
                m("src.monitoring.drift_exporter", "--once", "--window", "300"),
                [PY, "scripts/send_traffic.py", "--csv", "data/production/batch_drifted.csv", "--n", "300", "--url", url],
                m("src.monitoring.drift_exporter", "--once", "--window", "300"))
            run("09_api_session", [PY, "scripts/api_session.py", "--url", url, "--n", "200"])
        finally:
            server.terminate()
            server.wait(timeout=10)

    run("10_drift_normal", m("src.monitoring.run_drift_check", "--prod", "data/production/batch_normal.csv"))
    run("11_drift_shifted",
        m("src.monitoring.run_drift_check", "--prod", "data/production/batch_drifted.csv"),
        m("src.monitoring.run_drift_check", "--prod", "data/production/batch_drifted.csv", "--labels-available"))
    run("17_perf_monitor", m("src.monitoring.perf_monitor", "--prod", "data/production/batch_normal.csv"),
        m("src.monitoring.perf_monitor", "--prod", "data/production/batch_drifted.csv"))
    run("12_retrain_promotion",
        m("src.models.retrain"),
        m("src.models.retrain", "--labels-available"),
        m("src.models.retrain", "--labels-available", "--approve-as", "A. Rao (Model owner), with Risk and Compliance sign-off"),
        m("src.models.retrain", "--rollback"))
    print("done. Now run:  python scripts/build_results.py")


if __name__ == "__main__":
    main()
