"""Score a production CSV IN-PROCESS and append every prediction to logs/predictions.jsonl (same format as the API).

WHY: lets you test the drift exporter and the dashboards without starting the API.  (To exercise the real HTTP path use
scripts/send_traffic.py instead.)

python -m src.monitoring.simulate --csv data/production/batch_drifted.csv --n 500
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.paths import CHAMPION_PATH, PREDICTION_LOG
from src.data.make_dataset import REQUEST_FIELDS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    art = load_artifact(CHAMPION_PATH)
    df = pd.read_csv(a.csv).head(a.n)
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            req = P.PredictRequest(**{k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k])) for k in REQUEST_FIELDS})
            frame = P.request_to_frame(req)
            p = float(art["pipeline"].predict_proba(frame[art["input_columns"]])[:, 1][0])
            rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **frame.iloc[0].to_dict(),
                   "probability": round(p, 6), "risk_level": P.risk_level(p), "model_version": art["version"]}
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"appended {len(df)} predictions to {PREDICTION_LOG}")
