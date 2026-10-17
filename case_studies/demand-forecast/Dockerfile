# ---- Serving image -------------------------------------------------------
# Build :  docker build -t demand-forecast:1.0 .
# Run   :  docker run -p 8000:8000 demand-forecast:1.0
# Test  :  python scripts/smoke_test.py --url http://localhost:8000
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 1) dependencies first -> Docker caches this layer until requirements change
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# 2) application code + the model artifact + reference data + history store
COPY src/ ./src/
COPY models/champion.joblib ./models/champion.joblib
COPY data/processed/reference.csv ./data/processed/reference.csv
# the HISTORY STORE (stand-in for an online feature store): lag / rolling features are built from it at request time
COPY data/processed/sales_history.csv ./data/processed/sales_history.csv
COPY configs/ ./configs/

# 3) run as a non-root user (basic container hardening)
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
