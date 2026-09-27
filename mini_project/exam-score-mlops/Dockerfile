# Dockerfile
#
# Packages the FastAPI application + the trained model artifact into a single,
# independently-runnable image (PHASE 13). It does NOT depend on the host's Python
# environment -- everything the app needs (Python runtime, code, model, dependencies)
# lives inside the image.
#
# IMPORTANT: the model must already be trained before building this image --
#     python -m src.train
# produces models/exam_score_pipeline.joblib, which this Dockerfile copies in. Building
# without it will succeed (Docker doesn't know the file is "supposed" to exist), but the
# container will fail fast and loudly at startup (see api/main.py's lifespan handler,
# and PHASE 28's "missing model file" error-handling requirement) rather than silently
# serving broken predictions.

FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr -- without this, `docker logs` can lag
# behind what's actually happening inside the container, which is confusing during a
# live demo.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies BEFORE copying application code -- Docker caches layers, so if only
# your code changes (not requirements.txt), `docker build` reuses the cached dependency
# layer instead of reinstalling everything from scratch every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the trained artifact
COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY params.yaml .

# Run as a non-root user -- a basic container security practice (PHASE 29), not strictly
# required for this to work, but a bad habit to skip even in a teaching project.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/monitoring /app/reports \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
