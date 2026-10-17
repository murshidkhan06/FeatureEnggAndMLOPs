# MCP server: exposes two tools (predict_exam_score, get_model_information) over streamable HTTP.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependencies first so Docker can cache this layer when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY tools/ tools/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Inside a container the server must listen on 0.0.0.0, otherwise other containers can't reach it.
ENV MCP_HOST=0.0.0.0 MCP_PORT=8001 MCP_TRANSPORT=streamable-http
EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3)" || exit 1

CMD ["python", "server.py"]
