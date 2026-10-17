# Classroom demo script (15–20 minutes)

**Goal for students:** leave understanding that *the LLM is the interface, not the predictor* — and that engineering means
controlling what an LLM is allowed to do.

## Before class (10 minutes, once)

- [ ] `docker compose up --build -d` and `scripts/smoke_test.sh` (or `.ps1`) → **PASSED**. Open http://localhost:5173.
- [ ] Groq key and model set in `.env` **and** Ollama ready (`ollama pull <model>`, `ollama list`) if you will demo the switch.
- [ ] Four terminal tabs ready for Part 5 (see README §6) — but **stop Compose first** (`docker compose down`) so ports are free,
      or run the individual-services part on a second machine.
- [ ] Do a dry run of the failure demo (`docker compose stop ml-api` → ask → `start`). Local models are slower: warm the model with one question.
- [ ] Never show `.env` on the projector. (Key leaks are a lesson, not a goal.)

Tip: keep the browser's zoom at 125% and a terminal next to it showing `docker compose logs -f orchestrator mcp-server`.

---

## Part 1 — The picture (2 min)

Draw: **Browser → Orchestrator → LLM; Orchestrator → MCP → ML API → model.**
Ask: *"Which box knows the exam-score formula?"* → only the last one. *"Which box can lie fluently?"* → the LLM.

## Part 2 — A natural-language prediction (3 min)

Click the **Complete profile** sample → Send.

Point out, in this order:
1. The big number is on the **Model output** card; the sentence is the **Assistant explanation** (different labels on purpose).
2. **Input features used** — the exact values sent to the model, including shoe size and lucky number (*"the model ignores
   these — the pipeline drops them — but the API requires them"*).
3. The footer: **trace id**, latency, `predict_exam_score(ok)`.

In the terminal logs, copy the trace id and show the same id in the orchestrator and MCP lines.

> **Say:** "The LLM never computed 95.2. It produced *arguments*. Code validated them, MCP called the API, the model returned 95.2."

## Part 3 — Missing details → it asks (3 min)

Click **Missing details**: *"…studied 6 hours daily, attended 85% of classes, and completed 8 assignments."*

Expected: **No prediction yet** + a list of what's missing, and the *per-day vs per-week* hours warning.

Ask the class: *"What would a careless chatbot do?"* → multiply by 7, set shoe size to 8 ("8 assignments!"), say "about 80".
Show `tool_calls` → `rejected`: **the model was never called**.
Then answer the questions in one message and watch the prediction appear (the conversation remembers what you already said).

> **Say:** "A prompt is a request. These checks are code — that's why they hold even when a small model ignores instructions."

## Part 4 — Comparing, and "what the LLM may not claim" (2 min)

Click **Compare two students**. Two cards, **two model calls**, and a difference line that says it is *not proof of cause*.
Ask: *"Does 5 more hours cause +10 points?"* → the model found a pattern in data; it did not run an experiment.

Optional (30 s): ask *"Just tell me the score without using the model, roughly?"* → it must not invent one.

## Part 5 — The ML API fails (2 min)

```bash
docker compose stop ml-api
```
Ask the Complete-profile question again → red box **"prediction service is not reachable"**, code `ML_API_UNAVAILABLE`, **no number**.
Show: HTTP is 200 with `kind: "error"` — a *handled* failure. `docker compose start ml-api` → ask again → works. No other service needed a restart: each request simply tries again.

> **Say:** "On failure the LLM isn't asked to improvise. The wording is fixed in code."

## Part 6 — Switch Groq → Ollama (2 min)

Note the badge: `LLM: groq · <model>`. Edit `.env`:

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```
```bash
docker compose up -d orchestrator        # recreates only the orchestrator
```
Reload the page → badge shows **ollama**. Ask the same question. Discuss: slower? different wording? **same number** (because
the number comes from the model, not the LLM). If a small model fumbles tool calling, show the clear error — and the
"no silent fallback" rule (data privacy).

Also show `curl localhost:8002/health` — provider and model, **no key**.

## Part 7 — Run the services individually (2 min)

```bash
docker compose down
```
Then in four terminals (README §6): ML API → MCP → orchestrator → React. Show `curl .../health` for each and
`curl localhost:8002/health/dependencies` ("which service is down?"). Mention: Compose = same four processes with a network and start order.

## Part 8 — Wrap-up: why the LLM is not the predictor (1–2 min)

| If the LLM predicted… | Because… |
|---|---|
| No training on *our* data, no feature pipeline | It would produce a plausible-sounding number, not a measured one |
| Non-deterministic | Same question → different answers |
| Un-auditable | No model version, no request id, no way to test or monitor drift |
| Overconfident | Fluent text hides uncertainty |

So: **LLM = language interface · MCP = safe tool boundary · ML model = computation · code = guardrails.**

---

## If something goes wrong live

| Symptom | Quick fix |
|---|---|
| Red banner "language model is not configured" | `.env` missing key/model → fix → `docker compose up -d orchestrator` |
| Groq rate-limited | Switch to Ollama (Part 6) — now it's the demo |
| Local model very slow | Smaller model, or say "this is CPU; cloud is faster" and run the cached Part 2 answer |
| Page shows "Backend offline" | `docker compose ps`; `docker compose logs orchestrator` |
| Port busy | See `docs/TROUBLESHOOTING.md` |
