"""Static checks for everything we cannot RUN on a laptop without Docker / Kubernetes (run it before `docker build`).

    python scripts/static_checks.py            # needs models/champion.joblib and the data/processed files (after training)

  1. every .yaml / .yml / .json file parses
  2. every `COPY <source>` in the Dockerfile exists (after data generation + training)
  3. the Docker CMD module imports WITHOUT the development-only packages (mlflow, yaml, shap, matplotlib, dvc, requests, pytest ...)
     -> proves requirements-serving.txt is enough for the image
  4. Kubernetes: Service selector == Deployment pod labels; container image == the image name used in the docs; probes point at /health
  5. docker-compose: the services reference existing files; Prometheus targets match the compose service names
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import importlib
import json
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
SLUG, IMAGE = "credit-scoring", "credit-scoring:1.0"
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)


# 1 --- parse every config file
skip = {".venv", "mlruns", ".dvc", "__pycache__", "node_modules"}
n = 0
for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix in (".yaml", ".yml", ".json", ".ipynb") and not (skip & set(p.relative_to(ROOT).parts)) and "reports" not in p.parts:
        if p.name == "dvc.lock":
            continue
        try:
            (json.loads if p.suffix in (".json", ".ipynb") else yaml.safe_load)(p.read_text(encoding="utf-8"))
            n += 1
        except Exception as exc:  # noqa: BLE001
            check(False, f"{p.relative_to(ROOT)} does not parse: {exc}")
check(True, f"{n} yaml/json/ipynb files parse")

# 2 --- Dockerfile COPY sources
docker = (ROOT / "Dockerfile").read_text()
for src in re.findall(r"^COPY\s+(\S+)\s+\S+", docker, flags=re.M):
    check((ROOT / src).exists(), f"Dockerfile COPY source exists: {src}")
check(f"{SLUG}" in docker, "Dockerfile header names the image")

# 3 --- CMD module imports with dev-only packages blocked
cmd = re.search(r'^CMD \[(.*)\]', docker, flags=re.M).group(1)
module = re.search(r'"([\w.]+):app"', cmd).group(1)
BLOCKED = {"mlflow", "yaml", "shap", "matplotlib", "seaborn", "dvc", "requests", "httpx", "pytest"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not installed in the serving image")
        return None


sys.meta_path.insert(0, Blocker())
for k in [k for k in sys.modules if k.split(".")[0] in BLOCKED]:
    del sys.modules[k]
for m in (module, "src.monitoring.drift_exporter"):
    try:
        importlib.import_module(m)
        check(True, f"{m} imports without dev-only packages")
    except ImportError as exc:
        check(False, f"{m}: {exc}")
sys.meta_path.pop(0)
serving = (ROOT / "requirements-serving.txt").read_text().lower()
for pkg in ("fastapi", "uvicorn", "pydantic", "prometheus-client", "pandas", "numpy", "scikit-learn", "scipy", "joblib"):
    check(pkg in serving, f"requirements-serving.txt lists {pkg}")

# 4 --- Kubernetes consistency
dep = yaml.safe_load((ROOT / "k8s/deployment.yaml").read_text())
svc = yaml.safe_load((ROOT / "k8s/service.yaml").read_text())
labels = dep["spec"]["template"]["metadata"]["labels"]
check(svc["spec"]["selector"].items() <= labels.items(), f"Service selector {svc['spec']['selector']} matches pod labels {labels}")
check(dep["spec"]["selector"]["matchLabels"].items() <= labels.items(), "Deployment selector matches pod labels")
ctr = dep["spec"]["template"]["spec"]["containers"][0]
check(ctr["image"] == IMAGE, f"Deployment image is {ctr['image']}")
check(ctr["readinessProbe"]["httpGet"]["path"] == "/health" and ctr["livenessProbe"]["httpGet"]["path"] == "/health", "probes use /health")
check(svc["spec"]["ports"][0]["targetPort"] == ctr["ports"][0]["containerPort"], "Service targetPort == containerPort")
check(dep["metadata"]["name"] == SLUG and svc["metadata"]["name"] == SLUG, "k8s object names use the project slug")

# 5 --- compose + prometheus
comp = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
prom = yaml.safe_load((ROOT / "monitoring/prometheus.yml").read_text())
targets = [t for job in prom["scrape_configs"] for sc in job["static_configs"] for t in sc["targets"]]
check({"api:8000", "drift-exporter:9100"} <= set(targets) and {"api", "drift-exporter"} <= set(comp), "Prometheus targets match compose service names")
for s in comp.values():
    for v in s.get("volumes", []):
        host = v.split(":")[0]
        if host.startswith("./"):
            check((ROOT / host).exists() or host.startswith("./logs"), f"compose volume source exists: {host}")
check((ROOT / ".github/workflows/ci.yml").exists() and SLUG in (ROOT / ".github/workflows/ci.yml").read_text(), "CI workflow exists and uses the project name")
mlp = yaml.safe_load((ROOT / "MLproject").read_text())
check(mlp["name"] == SLUG and {"train", "drift", "retrain"} <= set(mlp["entry_points"]), "MLproject name + entry points")
check("PROJECT_NAME" not in "".join(p.read_text() for p in [ROOT / "Dockerfile", ROOT / "MLproject", ROOT / "k8s/deployment.yaml", ROOT / "k8s/service.yaml"]),
      "no PROJECT_NAME / PROJECT_IMAGE placeholders left")
print("\nSTATIC CHECKS:", "ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
