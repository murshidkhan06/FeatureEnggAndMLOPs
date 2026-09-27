# Makefile
#
# Convenience wrappers around the commands in the README. Every target here is also
# documented as a plain command students can run by hand -- `make` is a shortcut, not a
# requirement. Works on Linux/macOS out of the box; Windows students can either run the
# underlying commands directly (see README) or use `make` via WSL/Git Bash.

PYTHON := python
DOCKER_IMAGE := exam-score-api
DOCKER_TAG := 1.0

.PHONY: help install data pipeline train test api docker-build docker-run drift retrain monitor mlflow-ui clean

help:
	@echo "Targets: install data pipeline train test api docker-build docker-run drift retrain monitor mlflow-ui clean"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) data/raw/generate_data.py

# The full ML TRAINING pipeline (validate -> train -> evaluate -> mlflow -> test).
# Deliberately does NOT touch Docker -- see src/pipeline.py's docstring for why.
pipeline:
	$(PYTHON) -m src.pipeline

train:
	$(PYTHON) -m src.train

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

drift:
	$(PYTHON) -m src.drift

retrain:
	$(PYTHON) -m src.retrain

monitor:
	$(PYTHON) -m src.monitoring

# ---- APPLICATION DEPLOYMENT pipeline (separate from training, on purpose) -------------
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
