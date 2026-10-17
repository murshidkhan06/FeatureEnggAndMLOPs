# Troubleshooting

**First move, always:** `curl http://localhost:8002/health/dependencies` (or `scripts/smoke_test.sh`). It says *which* of
LLM / MCP server / ML API is unhealthy. Then read that service's log (`docker compose logs --tail 50 <service>`).

## Quick map

| Symptom | Most likely cause | Fix |
|---|---|---|
| UI banner "Cannot reach the orchestrator" | Orchestrator not running, or `VITE_API_BASE_URL` wrong | Start it; `curl localhost:8002/health`; in Docker rebuild the frontend if you changed the URL |
| Browser console: *CORS policy … blocked* | UI served from an origin not in `CORS_ALLOW_ORIGINS` (e.g. `127.0.0.1` vs `localhost`, or another port) | Add the exact origin (scheme + host + port) to `CORS_ALLOW_ORIGINS`, restart the orchestrator |
| Banner "GROQ_API_KEY is not set" / `LLM_NOT_CONFIGURED` | `.env` not read, or empty | Compose: `.env` next to `docker-compose.yml`; then `docker compose up -d orchestrator`. Local: `orchestrator/.env`, restart |
| `GROQ_MODEL is not set` | No model chosen on purpose | Set a tool-capable model from the Groq console |
| `LLM_AUTH_FAILED` | Wrong/expired key (or stray spaces/quotes in `.env`) | Re-create the key; no quotes needed: `GROQ_API_KEY=gsk_…` |
| `LLM_MODEL_NOT_FOUND` | Model id typo / retired / not pulled | Check the provider's list; Ollama: `ollama pull <model>` and `ollama list` |
| `LLM_RATE_LIMITED` | Free-tier limits | Wait; fewer questions/minute; or switch to Ollama |
| `LLM_TOOLS_UNSUPPORTED` | The model has no tool-calling | Use a tool-capable model (e.g. a Llama 3.1 / Qwen 2.5 class) |
| `LLM_INVALID_TOOL_ARGS` / `LLM_TOO_MANY_STEPS` | Small model misuses tools | Rephrase, or use a stronger model — a good lesson, not a bug |
| `LLM_TIMEOUT` | Local model on CPU is slow | Smaller model; raise `LLM_TIMEOUT_SECONDS`; warm it with one question first |
| `LLM_UNAVAILABLE` with Ollama | Ollama not running, or wrong URL | `ollama serve`; `curl http://localhost:11434/api/tags`; Docker: see below |
| `MCP_UNAVAILABLE` | MCP server down, wrong `MCP_SERVER_URL`, or Host not allowed | `curl localhost:8001/health`; Docker: `http://mcp-server:8001/mcp`, never localhost |
| Answer says "prediction service is not reachable" (`ML_API_UNAVAILABLE`) | ML API down / wrong `ML_API_BASE_URL` | `curl localhost:8000/health`; Docker: must be `http://ml-api:8000` |
| `ML_API_MODEL_NOT_LOADED` / ml-api container unhealthy | `models/exam_score_pipeline.joblib` missing | In `exam-score-mlops`: `python -m src.train`, then rebuild `ml-api` |
| `ML_API_REJECTED_INPUT` | A value outside the API's ranges got through | Read `error.details`; the API's `/docs` shows limits |
| Chatbot keeps asking for shoe size / lucky number | Working as designed: the API requires them | Provide any value (they are ignored by the model) |
| Chatbot refuses "6 hours daily" | Unit guard (model wants hours per **week**) | Give the weekly total |
| Score above 100 with a ⚠ notice | Model extrapolating outside its training data | Expected for extreme inputs — a teaching moment (Lab 5) |
| Nothing happens after Send | Previous request still running (button says "Working…") | Wait, or press **New conversation** |

## Docker-specific

**`docker compose config` shows an empty `GROQ_MODEL` or a value starting with `#`.**
Don't put a comment after a value in `.env` (`KEY=value # note`) — Docker Compose and Python read inline comments differently.
Put comments on their own line.

**`localhost` doesn't work between containers.** Inside a container `localhost` is the container itself. Use service names:
`ml-api`, `mcp-server`, `orchestrator`. Only your *browser* uses `localhost:<published port>`.

**Orchestrator can't reach Ollama running on my machine.**
* Docker Desktop (Windows/macOS): `OLLAMA_BASE_URL=http://host.docker.internal:11434` works as is.
* Linux: Ollama listens on 127.0.0.1 only, so containers can't connect. Run Ollama with `OLLAMA_HOST=0.0.0.0` (trusted network only), or use
  the Compose Ollama service: `docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d`.
* Test from inside the container: `docker compose exec orchestrator python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:11434/api/tags').read()[:100])"`.

**Ollama service is up but the model isn't there.** `docker compose exec ollama ollama pull llama3.1:8b`, then `docker compose exec ollama ollama list`.

**Changed `.env` but nothing changed.** Env vars are read at container creation: `docker compose up -d orchestrator` (recreates it).
The frontend URL is baked in at *build* time: `docker compose up -d --build frontend`.

**`port is already allocated`.** Something else uses 5173/8000/8001/8002 (often a leftover terminal run). Stop it, or change the left side of the
`ports:` mapping. If you change the UI port, add the new origin to `CORS_ALLOW_ORIGINS`.

**ml-api build fails: `exam-score-mlops` not found.** Set `ML_API_DIR` in `.env` to the project's real path.

**Container "unhealthy" / services never start.** `docker compose ps` then `docker compose logs <service>`. The start order is
ml-api → mcp-server → orchestrator → frontend; the first unhealthy one is the culprit.

**Slow first build.** Python wheels and `npm ci` download once; later builds use the layer cache. `docker compose build --no-cache <service>` forces a clean build.

**Windows line endings / file sharing errors.** Keep the project in a normal folder (not a OneDrive-synced one) and make sure Docker Desktop is set to the WSL2 backend.

## Local (non-Docker) specific

* `ModuleNotFoundError: app` → run `uvicorn app.main:app` **from the `orchestrator/` folder**. `server.py` runs from `mcp_server/`.
* `ExceptionGroup` / syntax errors → you are on Python < 3.11. Use 3.11+.
* `mcp` import errors about `FastMCP` → MCP SDK 2.x was installed; this project pins **`mcp==1.30.0`** (`pip install -r requirements.txt` inside the service's venv).
* PowerShell: `curl` is an alias of `Invoke-WebRequest`. Use `curl.exe` or `Invoke-RestMethod`.
* PowerShell can't activate the venv → `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
* Vite shows the old API URL → restart `npm run dev` after editing `frontend/.env`.
* Port 11434 refused → `ollama serve` (the desktop app starts it automatically on most systems).

## Reading the logs

Each line is JSON: `{"ts", "level", "service", "trace_id", "event", …}`. Useful events:

| Service | Event | Meaning |
|---|---|---|
| orchestrator | `llm_call` | One LLM request (`outcome`, `latency_ms`) |
| orchestrator | `tool_call_rejected` | Guard stopped a call (missing/unclear input) — model not called |
| orchestrator | `tool_call` | Tool ran or failed (`status`, `error_code`) |
| orchestrator | `assistant_text_replaced` | The LLM quoted a score the model never produced |
| orchestrator | `chat_turn_done` / `http_request` | End of the question |
| mcp-server | `predict_ok` / `predict_failed` | The ML API call (`ml_request_id`, feature **names**, never values) |

Find one question everywhere: `docker compose logs | grep <trace_id>`.

## Still stuck?

Collect: the `trace_id` from the UI, `docker compose ps`, and `docker compose logs --tail 80 <service>` (check for secrets before sharing — keys are redacted
in this project's logs, but your shell history is not).
