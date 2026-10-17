"""CI quality gate: fail the build if the champion model is below the minimum bar.
Edit MIN thresholds in configs/quality_gate.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import json
import sys
from pathlib import Path

from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, CONFIGS

gate = json.loads((Path(CONFIGS) / "quality_gate.json").read_text())
m = load_artifact(CHAMPION_PATH)["metrics"]
failed = [f"{k}: {m.get(k)} < {v}" for k, v in gate["min"].items() if m.get(k, 0) < v]
print("metrics:", m)
if failed:
    print("QUALITY GATE FAILED ->", failed)
    sys.exit(1)
print("QUALITY GATE PASSED")
