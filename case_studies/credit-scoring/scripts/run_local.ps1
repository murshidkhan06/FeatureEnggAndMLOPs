# Windows PowerShell:  .\scripts\run_local.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data.make_dataset
python -m src.features.build_features
python -m src.features.select_features
python -m src.models.train --bootstrap-champion
pytest -q
Write-Host "Start the API with:  uvicorn src.api.app:app --reload --port 8000"
