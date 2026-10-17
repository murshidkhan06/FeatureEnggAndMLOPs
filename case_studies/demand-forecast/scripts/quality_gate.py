"""CI quality gate: fail the build if the champion model is below the minimum bar (configs/quality_gate.json).

  min      metric must be >= value      roc_auc   how well the forecast RANKS spike days above normal days (0.5 = coin flip)
  max      metric must be <= value      wape      share of units sold that the forecast got wrong (lower is better)
  max_abs  |metric| must be <= value    bias_pct  systematic over/under-forecast as a share of units sold

Example: wape 0.1285 <= 0.16 passes;  bias_pct -0.023 -> |-0.023| = 0.023 <= 0.05 passes;  roc_auc 0.86 >= 0.70 passes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable when run as a script

import json

from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, CONFIGS

gate = json.loads((Path(CONFIGS) / "quality_gate.json").read_text())
m = load_artifact(CHAMPION_PATH)["metrics"]
failed = [f"{k}: {m.get(k)} < {v}" for k, v in gate.get("min", {}).items() if m.get(k) is None or m[k] < v]
failed += [f"{k}: {m.get(k)} > {v}" for k, v in gate.get("max", {}).items() if m.get(k) is None or m[k] > v]
failed += [f"|{k}|: {abs(m[k]) if k in m else None} > {v}" for k, v in gate.get("max_abs", {}).items() if k not in m or abs(m[k]) > v]
print("metrics:", {k: m.get(k) for k in [*gate.get("min", {}), *gate.get("max", {}), *gate.get("max_abs", {})]})
if failed:
    print("QUALITY GATE FAILED ->", failed)
    sys.exit(1)
print("QUALITY GATE PASSED")
