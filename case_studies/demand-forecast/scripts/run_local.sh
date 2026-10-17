#!/usr/bin/env bash
set -e
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.make_dataset
python -m src.features.select_features
python -m src.models.train --bootstrap-champion
pytest -q
python scripts/quality_gate.py
echo "Start the API with:  uvicorn src.api.app:app --reload --port 8000"
