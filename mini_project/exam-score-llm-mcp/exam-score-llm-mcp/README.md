# Natural-Language Exam Score Prediction — LLM + MCP + React + FastAPI

Ask a question in plain English ("Predict the score for a student who studies 12.5 hours a week, 88% attendance…")
and get a **real prediction from your trained scikit-learn model** — with the LLM used only to *understand the
question* and *explain the result*.

> **The one rule of this project:** the LLM is never the predictor. The number on screen always comes from the
> fitted model behind the existing ML API. If the LLM tries to invent a number or a missing value, code stops it.

This project **re-uses** the earlier `exam-score-mlops` project (its FastAPI `/predict` endpoint and fitted
pipeline). It does not retrain, copy or modify it. Details: [`ml_api/README.md`](ml_api/README.md) and
[`docs/ML_API_CONTRACT.md`](docs/ML_API_CONTRACT.md).

---

## 1. Architecture

```
┌──────────────┐  HTTP/JSON   ┌───────────────────────┐  MCP over HTTP   ┌──────────────┐  HTTP/JSON  ┌────────────────────┐
│  React UI    │ ───────────► │  LLM Orchestrator     │ ───────────────► │  MCP Server  │ ──────────► │  Existing ML API   │
│  (browser)   │ ◄─────────── │  FastAPI  :8002       │ ◄─────────────── │  :8001       │ ◄────────── │  FastAPI  :8000    │
│  :5173       │              │  + MCP client         │                  │  2 tools     │             │  fitted sklearn    │
└──────────────┘              └──────────┬────────────┘                  └──────────────┘             │  Pipeline (.joblib)│
                                         │ OpenAI-style /chat/completions                              └────────────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ Groq (cloud, default)│   switch with one line:  LLM_PROVIDER=ollama
                              │   or Ollama (local)  │
                              └──────────────────────┘
```

| Layer | Responsibility | What it must **never** do |
|---|---|---|
| **React UI** | Chat, show the prediction card and the inputs used, samples, reset, provider badge. | Call Groq/Ollama/MCP/ML API directly; hold any secret. |
| **Orchestrator** (FastAPI) | Owns the conversation. Asks the LLM what to do, **validates** every tool call, runs it through MCP, checks the LLM's wording, returns one consistent JSON shape. | Trust the LLM's arguments or its numbers. |
| **LLM** (Groq / Ollama) | Turns wording into structured tool arguments; writes a short explanation. | Invent a prediction or a missing value; claim causes. |
| **MCP server** | Exposes exactly two safe tools (`predict_exam_score`, `get_model_information`) with strict typed schemas; calls the ML API at one fixed address. | Accept a URL or any free-form instruction from the LLM. |
| **ML API + model** | Does all feature engineering (imputation, scaling, encoding, PCA, selection) inside the fitted pipeline and computes the score. | Be re-implemented anywhere else. |

**Why MCP in the middle?** The *Model Context Protocol* is a standard way to publish "tools" an AI app can call.
The MCP server is a small, auditable gate: the LLM can only do what the tools allow, and the tool schema is
published in one place and re-used for validation.

### One request, step by step

1. You type a sentence → React `POST /chat` with an `X-Request-ID` (the **trace id**).
2. The orchestrator sends your sentence + the tool list to the LLM.
3. The LLM answers *"call `predict_exam_score` with {…}"* (or asks nothing / answers off-topic).
4. **Code checks the arguments** (types, ranges, unknown fields, *did the user really say this?*, daily-vs-weekly hours).
   Anything doubtful → a clarification question; **the model is not called**.
5. Valid → the orchestrator calls the tool through MCP → MCP calls `POST /predict` on the ML API.
6. The real score comes back; the LLM writes a short explanation; code checks that explanation doesn't quote a score the
   model never produced.
7. React shows **Model output** (the card) separately from **Assistant explanation** (the wording).

---

## 2. Folder layout

```
exam-score-llm-mcp/
├── frontend/              React + TypeScript + Vite UI (+ Dockerfile, nginx.conf, vitest tests)
├── orchestrator/          FastAPI app: LLM providers, MCP client, guards, chat service (+ Dockerfile)
├── mcp_server/            MCP server (official Python SDK): tools, ML API client (+ Dockerfile)
├── ml_api/                README only — the ML API is the EXISTING project, built from ../exam-score-mlops
├── tests/                 pytest: unit tests (LLM + ML API mocked) + integration tests (real ML API)
├── scripts/               smoke tests (bash + PowerShell) and a fake Ollama for offline practice
├── docs/                  API examples, ML API contract, demo script, lab exercises, troubleshooting
├── docker-compose.yml     frontend + orchestrator + mcp-server + ml-api
├── docker-compose.ollama.yml   optional: run Ollama as a Compose service
├── .env.example           Compose settings (copy to .env)
└── README.md
```

Expected side-by-side layout on your machine:

```
some-folder/
├── exam-score-mlops/      ← your EXISTING ML project (unchanged)
└── exam-score-llm-mcp/    ← this project
```

---

## 3. Ports and prerequisites

| Service | Port | URL |
|---|---|---|
| React UI | 5173 | http://localhost:5173 |
| Orchestrator | 8002 | http://localhost:8002/docs |
| MCP server | 8001 | http://localhost:8001/health (MCP endpoint is `/mcp`) |
| ML API | 8000 | http://localhost:8000/docs |
| Ollama (optional) | 11434 | http://localhost:11434 |

You need: **Python 3.11+** (3.13 was used for testing), **Node.js 20+** (22 used), **Docker Desktop / Docker Engine with
Compose v2** (only for the Docker route), and either a **Groq API key** or **Ollama**.

---

## 4. Choose and configure the LLM

The code has **no hidden default model**. You choose one on purpose; otherwise `/chat` answers
`503 LLM_NOT_CONFIGURED` with a clear message (and the UI shows a banner).

**Model names change.** The examples below were reasonable choices when this was written — check the provider's
current model list and that the model supports **tool calling** before the lab.

### Option A — Groq (cloud, default)

1. Create a key at <https://console.groq.com> (free tier available; limits change).
2. Put it in a **git-ignored** `.env` file — never in code, never in the frontend, never in chat.
3. Set `LLM_PROVIDER=groq`, `GROQ_API_KEY=…`, `GROQ_MODEL=llama-3.3-70b-versatile` (example; verify).

### Option B — Ollama (local, no internet, no key)

```bash
ollama --version
ollama pull llama3.1:8b          # example; any tool-capable model works (e.g. qwen2.5:7b). Several GB.
ollama list                      # confirm it is downloaded
ollama serve                     # only if the Ollama app/service is not already running
curl http://localhost:11434/api/tags
```

Set `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.1:8b`. Small local models are **less reliable at tool calling** than
large cloud ones — that is a good classroom discussion, and exactly why the guards exist.

### Switching Groq ↔ Ollama

It is configuration only. Change `LLM_PROVIDER` (and the model) and restart the orchestrator. The orchestrator **never
silently falls back** to the other provider — a failure is reported, so student data never goes to a cloud service by surprise.

---

## 5. Run it — Route 1: Docker Compose (everything at once)

```bash
cd exam-score-llm-mcp
cp .env.example .env                 # Windows PowerShell:  Copy-Item .env.example .env
# edit .env: set LLM_PROVIDER, GROQ_API_KEY + GROQ_MODEL   (or OLLAMA_MODEL)
# if your ML project is not at ../exam-score-mlops, set ML_API_DIR in .env

docker compose config                # validate + show the final configuration (check no key is missing)
docker compose up --build            # build and start (add -d to run in the background)
```

Open **http://localhost:5173**. Services start in order using health checks:
`ml-api → mcp-server → orchestrator → frontend`.

| Task | Command |
|---|---|
| Status and health | `docker compose ps` |
| Follow all logs | `docker compose logs -f` |
| Follow one service | `docker compose logs -f orchestrator` |
| Rebuild + restart one service | `docker compose up -d --build orchestrator` |
| Restart without rebuilding | `docker compose restart mcp-server` |
| Stop one service (demo a failure) | `docker compose stop ml-api` / `docker compose start ml-api` |
| Apply a changed `.env` (e.g. new LLM) | `docker compose up -d orchestrator` |
| Frontend API URL changed | `docker compose up -d --build frontend` (it is baked in at build time) |
| Stop and remove containers | `docker compose down` |
| …and remove the Ollama model volume | `docker compose down -v` (re-downloads models!) |

**Networking in Docker (the #1 source of confusion):**
inside a container, `localhost` means *that container*. Containers reach each other by **service name**
(`http://ml-api:8000`, `http://mcp-server:8001/mcp`) — Compose sets these for you. Your **browser** is *not* in Docker, so it
uses the published ports (`http://localhost:8002`).

### Ollama with Docker — two ways

**(a) Ollama on your host machine (default, usually faster).** Keep `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
`docker-compose.yml` adds `host.docker.internal` for Linux too.
*Linux gotcha:* Ollama listens only on `127.0.0.1` by default, which containers cannot reach. Start it with
`OLLAMA_HOST=0.0.0.0` (for a systemd service: `sudo systemctl edit ollama` → `Environment="OLLAMA_HOST=0.0.0.0"`), and
only on a trusted network — this exposes Ollama to your LAN.

**(b) Ollama as a Compose service:**

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build -d
docker compose exec ollama ollama pull llama3.1:8b
# .env:  LLM_PROVIDER=ollama   OLLAMA_MODEL=llama3.1:8b
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d orchestrator
```

---

## 6. Run it — Route 2: each service on its own (best for learning and debugging)

Open **four terminals**. Start from the folder that contains both projects.

### Terminal 1 — ML API (the existing project)

```bash
cd exam-score-mlops
python -m venv .venv && source .venv/bin/activate        # Windows:  py -3 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ls models/exam_score_pipeline.joblib                      # must exist; if not:  python -m src.train
uvicorn api.main:app --port 8000
```
Check: `curl http://localhost:8000/health` → `{"status":"ok","model_loaded":true,...}`

### Terminal 2 — MCP server

```bash
cd exam-score-llm-mcp/mcp_server
python -m venv .venv && source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                      # defaults already point at localhost:8000
python server.py
```
Check: `curl http://localhost:8001/health`

### Terminal 3 — Orchestrator

```bash
cd exam-score-llm-mcp/orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                      # then edit: provider, key, model
uvicorn app.main:app --reload --port 8002
```
Check: `curl http://localhost:8002/health` (shows provider/model, **never** the key) and
`curl http://localhost:8002/health/dependencies` (checks LLM + MCP + ML API in one go).

### Terminal 4 — React UI

```bash
cd exam-score-llm-mcp/frontend
cp .env.example .env                                      # VITE_API_BASE_URL=http://localhost:8002
npm install
npm run dev
```
Open http://localhost:5173.

> **Windows PowerShell notes.** If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
> Use `Copy-Item` instead of `cp`, and `curl.exe` (not the `curl` alias) or `Invoke-RestMethod`.

---

## 7. Test it from the command line

**bash / macOS / Linux**

```bash
curl -s http://localhost:8002/chat -H "Content-Type: application/json" -d '{
  "message": "Predict the exam score for a student who studies 12.5 hours per week, has 88% attendance, scored 72, 75 and 70 in the three mock tests, belongs to the Medium income bracket, lives in Pune, enrolled on 2025-06-01, has shoe size 9 and lucky number 42."
}' | python -m json.tool
```

**Windows PowerShell**

```powershell
$body = @{ message = "Predict the exam score for a student who studies 12.5 hours per week, has 88% attendance, scored 72, 75 and 70 in the three mock tests, belongs to the Medium income bracket, lives in Pune, enrolled on 2025-06-01, has shoe size 9 and lucky number 42." } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8002/chat -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8
```

One-shot checks of every service: `scripts/smoke_test.sh` (bash) or `scripts/smoke_test.ps1` (PowerShell).
More requests and the exact responses (prediction, clarification, comparison, errors): [`docs/API.md`](docs/API.md).

---

## 8. Automated tests

```bash
cd exam-score-llm-mcp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                      # unit tests: LLM and ML API are mocked — no key, no GPU, no network
pytest -m integration       # also needs the REAL ML API running on :8000 (auto-skipped if it is down)

cd frontend && npm test     # React tests (vitest + Testing Library); also: npm run typecheck && npm run build
```

What the tests cover: tool schemas and ML-API error mapping; the guards (missing/invented values, unit confusion,
ungrounded scores); provider error mapping for Groq and Ollama; HTTP status codes, CORS, size limit, trace ids, secret
leakage; config files (`.env.example`, compose, Dockerfiles); an end-to-end chain with a real MCP server process and the
real ML API; and the UI (loading, errors, clarification, reset, exact values on the card).

---

## 9. What the code guarantees (and what it cannot)

| Risk | Defence (in code, not in the prompt) |
|---|---|
| LLM invents a prediction | The score on screen is only ever the ML API's number. If the explanation quotes a score the model never produced, it is **replaced** with a plain summary and a notice is shown. |
| LLM invents a missing value (e.g. defaults `shoe_size=8`) | Every value must be traceable to the user's own words (**provenance check**) *and* its topic must have been mentioned (**cue words**). Otherwise → clarification, no model call. |
| LLM invents a field (`assignments`) | Unknown fields are rejected. |
| "6 hours **daily**" vs the model's hours **per week** | Unit guard → asks for the weekly total instead of multiplying silently. |
| LLM sends text for a number, out-of-range values, bad dates | Validated against the schema the MCP server itself publishes. |
| LLM picks the URL / calls other things | No tool accepts a URL; only two named tools are allowed; the ML API address is one fixed setting. |
| Runaway tool loops | Max rounds and max predictions per question. |
| Model returns an impossible score (e.g. 103.3) | Shown unchanged **with a warning** (never silently clipped). |
| ML API down / slow / rejects input | A deterministic message — the LLM is not asked to improvise around a failure. |
| Secrets | Key lives only in the orchestrator's environment (`SecretStr`), is redacted in logs, never returned by any endpoint, never in the browser. |
| Causal claims ("studying more *causes* higher marks") | System prompt forbids it and the UI labels comparisons "two model outputs, not causal". |

**Honest limits** (great discussion topics):

* The provenance and cue checks are *keyword-level*. They stop the common failures but cannot understand meaning (for
  example a number word like "seventy" is treated as "not found" — a safe failure that just asks again).
* The model is a regression: it can **extrapolate** outside its training data (training study hours were ≈ 0.7–21.7 yet the
  API allows up to 40). The warning covers impossible scores only, not "plausible but unreliable" ones.
* The ML API *requires* `shoe_size` and `lucky_number` although the pipeline ignores them (deliberately irrelevant
  columns). The chatbot therefore asks for them — it does not default them.
* Conversation memory is in-process RAM (one worker, lost on restart). A real system needs Redis/a database and a data-retention policy.
* Small Ollama models may fail to produce valid tool calls; the app then returns a clear error rather than guessing.

---

## 10. Logging and the trace id

Every service writes **one JSON line per event**. The browser creates `X-Request-ID: web-<hex>`; it is returned as `trace_id`
and appears in the orchestrator and MCP logs. Logs contain **feature names, never values**, and never any key.

```bash
docker compose logs orchestrator mcp-server ml-api | grep web-1a2b3c4d5e6f
```

The existing ML API ignores incoming request ids and creates its own; the MCP log line `predict_ok` records both
`trace_id` and `ml_request_id`, so you can follow the chain: **UI → orchestrator → MCP → (ml_request_id) → ML API log**.

---

## 11. What was verified, and what was not

Verified while building (cloud sandbox): all pytest suites (unit + integration against the **real** ML API and a **real** MCP
server process), the React tests, TypeScript type-check and production build, `docker compose config`, a run of the three
Python/Node services from clean copies of exactly the files each Dockerfile copies, the full chain over HTTP with a stand-in
Ollama (`scripts/fake_ollama.py`), and the ML-API-down path.

**Not verified here — please test on your machine first:** building the Docker images (the sandbox could not pull base
images from Docker Hub), **real Groq calls**, and **real Ollama models**. The provider code is tested against mocked
OpenAI-compatible responses. Run `docker compose up --build` and the smoke test once before class.

---

## 12. More

* [`docs/API.md`](docs/API.md) — endpoints, request/response examples, error codes
* [`docs/ML_API_CONTRACT.md`](docs/ML_API_CONTRACT.md) — the existing ML API's contract and how it is used
* [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — 15–20 minute classroom demo
* [`docs/LAB_EXERCISES.md`](docs/LAB_EXERCISES.md) — student labs
* [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — symptoms → causes → fixes
