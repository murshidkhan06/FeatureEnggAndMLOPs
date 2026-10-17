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

SNAPSHOTS_PATH = DATA_PROCESSED / "snapshots.csv"          # one row per (customer, as_of_date) + target + split
FEATURE_REGISTRY_PATH = CONFIGS / "feature_registry.yaml"   # feature metadata + lifecycle history
SELECTED_FEATURES_PATH = CONFIGS / "selected_features.yaml" # output of the feature-selection step

for _p in (DATA_RAW, DATA_PROCESSED, DATA_PRODUCTION, MODELS, LOGS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)
