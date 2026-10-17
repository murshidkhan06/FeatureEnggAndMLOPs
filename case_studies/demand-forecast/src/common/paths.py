"""Central place for file locations so every script agrees (Windows + Linux safe)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PRODUCTION = ROOT / "data" / "production"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"

CHAMPION_PATH = MODELS / "champion.joblib"          # the model the API serves
CHALLENGER_PATH = MODELS / "challenger.joblib"      # a newly trained candidate
REGISTRY_PATH = MODELS / "registry.json"            # tiny local registry (history of promotions)
PREDICTION_LOG = LOGS / "predictions.jsonl"         # every served prediction (for monitoring)
REFERENCE_PATH = DATA_PROCESSED / "reference.csv"   # reference distribution for drift checks
HISTORY_PATH = DATA_PROCESSED / "sales_history.csv"  # the "online feature store" stand-in: daily sales history the API reads
HISTORY_DRIFTED_PATH = DATA_PROCESSED / "sales_history_drifted.csv"  # same, but for the trend-shift scenario
FEATURES_PATH = DATA_PROCESSED / "features.csv"      # engineered training frame (output of build_features)

for _p in (DATA_RAW, DATA_PROCESSED, DATA_PRODUCTION, MODELS, LOGS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)
