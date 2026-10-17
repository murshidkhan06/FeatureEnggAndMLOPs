# Windows PowerShell:  .\scripts\run_local.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data.make_dataset                  # 1. raw tables
python -m src.features.build_features            # 2. leakage-safe monthly snapshots + production batches
python -m src.features.select_features           # 3. choose features on TRAINING rows only
python -m src.models.train --bootstrap-champion  # 4. baseline + candidate, MLflow, champion v1
pytest -q                                        # 5. tests
python scripts/quality_gate.py                   # 6. minimum bar
Write-Host "Start the API with:  uvicorn src.api.app:app --reload --port 8000"
