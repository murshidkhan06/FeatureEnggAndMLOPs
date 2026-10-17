# Orchestrator API

Base URL (local): `http://localhost:8002` · Interactive docs: `http://localhost:8002/docs`

The browser talks **only** to this API. Every response uses the same shapes, so the UI never has to guess.
IDs and timings below are real captured values from a run of the full stack; yours will differ.
`provider.model` is whatever you configured.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Ask a question (the main endpoint) |
| `DELETE` | `/chat/{conversation_id}` | Forget a conversation (`204`, even if it did not exist) |
| `GET` | `/health` | Liveness + provider/model + whether the LLM is configured (never a secret) |
| `GET` | `/health/dependencies` | Deeper check of LLM, MCP server and ML API |

## Request

```json
{ "message": "…up to 1000 characters…", "conversation_id": "optional – send back the id you were given" }
```

Optional header `X-Request-ID: web-<hex>` (8–64 chars `A–Z a–z 0–9 _ -`) becomes the **trace id**; otherwise one is generated.
It is returned in the `trace_id` field and in the `X-Trace-Id` response header.

## Response shape (`kind` tells the UI what to render)

| `kind` | Meaning | Rendered as |
|---|---|---|
| `prediction` | The model ran. `predictions[]` holds the real numbers. | Prediction card(s) + explanation |
| `clarification` | Something is missing/unclear. **The model was not called.** | "More information needed" list |
| `information` | Answer to "what does the model need?" from the MCP info tool. | Explanation + model information |
| `text` | Off-topic or small talk. **No tool was called.** | Explanation only |
| `error` | A tool/ML-API failure. HTTP is still `200`; `error` explains it. | Error box |

Other fields: `assistant_message` (wording), `tool_calls[]` (audit trail), `notices[]` (warnings, e.g. a replaced explanation or an
out-of-range score), `provider`, `latency_ms`.

> **Display rule for UI authors:** show `predictions[i].predicted_final_score` (from the model). Never parse a score out of
> `assistant_message`.

---

## 1. A complete profile → real prediction

```bash
curl -s http://localhost:8002/chat -H "Content-Type: application/json" -H "X-Request-ID: web-0123456789ab" \
  -d '{"message": "Predict the exam score for a student who studies 12.5 hours per week, has 88% attendance, scored 72, 75 and 70 in the three mock tests, belongs to the Medium income bracket, lives in Pune, enrolled on 2025-06-01, has shoe size 9 and lucky number 42."}'
```

```json
{
  "trace_id": "web-0123456789ab",
  "conversation_id": "22009156b44141c3aa46cc79c9b1b25e",
  "kind": "prediction",
  "assistant_message": "The model's predicted final score is 95.2. This is the model's estimate, not a guarantee.",
  "predictions": [
    {
      "label": "Student",
      "predicted_final_score": 95.2,
      "features": {
        "study_hours": 12.5,
        "attendance_pct": 88.0,
        "mock_test_1": 72.0,
        "mock_test_2": 75.0,
        "mock_test_3": 70.0,
        "income_bracket": "Medium",
        "city": "Pune",
        "enrollment_date": "2025-06-01",
        "shoe_size": 9.0,
        "lucky_number": 42
      },
      "model_version": "1792163631093186959",
      "ml_request_id": "52cdbe44-6b37-4c57-b763-2af8ae7a7f06"
    }
  ],
  "clarification": null,
  "model_information": null,
  "tool_calls": [
    {
      "tool": "predict_exam_score",
      "status": "ok",
      "label": "Student",
      "error_code": null,
      "latency_ms": 33.9
    }
  ],
  "error": null,
  "notices": [],
  "provider": {
    "name": "groq",
    "model": "<your GROQ_MODEL>"
  },
  "latency_ms": 140.9
}
```

## 2. Missing / unclear details → clarification (no model call)

Message: *"Predict the exam score for a student who studied 6 hours daily, attended 85% of classes, and completed 8 assignments."*
(Here the LLM tried to fill in the gaps; the code refused. The `rejected` tool call proves the model was never invoked.)

```json
{
  "trace_id": "web-0123456789ab",
  "conversation_id": "8a7a1a976b684a4c9c2a1358cd3fc0dd",
  "kind": "clarification",
  "assistant_message": "I can't run the prediction model yet, because I don't want to guess any values.\n\nSo far I have: attendance_pct = 85.\nI still need:\n  - mock_test_1: Mock test score out of 100.\n  - mock_test_2: Mock test score out of 100.\n  - mock_test_3: Mock test score out of 100.\n  - income_bracket: Family income bracket: Low, Medium or High.\n  - city: City name, e.g. Pune, Delhi, Mumbai, Bengaluru.\n  - enrollment_date: Enrollment date in YYYY-MM-DD format.\n  - shoe_size: Shoe size. (The model ignores this, but the ML API requires it.)\n  - lucky_number: Lucky number. (The model ignores this, but the ML API requires it.)\nThese need to be checked:\n  - study_hours: you described hours per DAY, but this field is hours per WEEK - please tell me the weekly total\n\nPlease send the missing or corrected values and I will run the model.",
  "predictions": [],
  "clarification": [
    {
      "label": "Student",
      "missing_fields": [
        "mock_test_1",
        "mock_test_2",
        "mock_test_3",
        "income_bracket",
        "city",
        "enrollment_date",
        "shoe_size",
        "lucky_number"
      ],
      "problems": [
        {
          "field": "study_hours",
          "problem": "you described hours per DAY, but this field is hours per WEEK - please tell me the weekly total"
        }
      ],
      "understood_values": {
        "attendance_pct": 85
      }
    }
  ],
  "model_information": null,
  "tool_calls": [
    {
      "tool": "predict_exam_score",
      "status": "rejected",
      "label": "Student",
      "error_code": "NEEDS_CLARIFICATION",
      "latency_ms": null
    }
  ],
  "error": null,
  "notices": [],
  "provider": {
    "name": "groq",
    "model": "<your GROQ_MODEL>"
  },
  "latency_ms": 105.0
}
```

## 3. Comparing two profiles (one model call per profile)

Message: *Compare two students. Student A: 10 hours per week, 80% attendance, mock tests 60, 65, 62, Low income, Delhi, enrolled 2025-01-15, shoe size 8, lucky number 7. Student B: same as A but 15 hours per week.*

```json
{
  "trace_id": "web-0123456789ab",
  "conversation_id": "f2195efc5f8c4a15afac0edda14d1f38",
  "kind": "prediction",
  "assistant_message": "Student B's model output is higher than Student A's. This compares two predictions only; it does not show the extra hours caused the difference.",
  "predictions": [
    {
      "label": "Profile 1",
      "predicted_final_score": 83.1,
      "features": {
        "study_hours": 10.0,
        "attendance_pct": 80.0,
        "mock_test_1": 60.0,
        "mock_test_2": 65.0,
        "mock_test_3": 62.0,
        "income_bracket": "Low",
        "city": "Delhi",
        "enrollment_date": "2025-01-15",
        "shoe_size": 8.0,
        "lucky_number": 7
      },
      "model_version": "1792163631093186959",
      "ml_request_id": "783da0df-fd5f-4462-9f0a-2d4875eb9ac5"
    },
    {
      "label": "Profile 2",
      "predicted_final_score": 93.2,
      "features": {
        "study_hours": 15.0,
        "attendance_pct": 80.0,
        "mock_test_1": 60.0,
        "mock_test_2": 65.0,
        "mock_test_3": 62.0,
        "income_bracket": "Low",
        "city": "Delhi",
        "enrollment_date": "2025-01-15",
        "shoe_size": 8.0,
        "lucky_number": 7
      },
      "model_version": "1792163631093186959",
      "ml_request_id": "05ba5274-90b4-483c-81b1-599e24e14f89"
    }
  ],
  "clarification": null,
  "model_information": null,
  "tool_calls": [
    {
      "tool": "predict_exam_score",
      "status": "ok",
      "label": "Profile 1",
      "error_code": null,
      "latency_ms": 32.3
    },
    {
      "tool": "predict_exam_score",
      "status": "ok",
      "label": "Profile 2",
      "error_code": null,
      "latency_ms": 33.7
    }
  ],
  "error": null,
  "notices": [],
  "provider": {
    "name": "groq",
    "model": "<your GROQ_MODEL>"
  },
  "latency_ms": 177.8
}
```

The UI shows both cards and (for exactly two) the arithmetic difference with the reminder that **two model outputs are not proof of cause and effect**.

## 4. "What does the model need?" → information

```json
{
  "trace_id": "web-0123456789ab",
  "conversation_id": "984c1450a43b492385d733a0228e2de5",
  "kind": "information",
  "assistant_message": "The model takes 10 inputs. Study hours is per week, between 0 and 40.",
  "predictions": [],
  "clarification": null,
  "model_information": {
    "ok": true,
    "api_title": "Student Exam Score Predictor",
    "api_description": "Predicts a student's final exam score from raw, unprocessed profile data. All feature engineering (imputation, scaling, encoding, PCA, feature selection) happens automatically inside the loaded pipeline.",
    "api_version": "1.0.0",
    "prediction_endpoint": "POST /predict",
    "target": "predicted_final_score (the ML API does not publish a unit or range for it)",
    "model_loaded": true,
    "model_version": "1792163631093186959",
    "input_fields": [
      {
        "name": "study_hours",
        "type": "number",
        "required": true,
        "minimum": 0.0,
        "maximum": 40.0,
        "description": "Weekly study hours"
      },
      {
        "name": "attendance_pct",
        "type": "number",
        "required": true,
        "minimum": 0.0,
        "maximum": 100.0,
        "description": "Attendance percentage"
      },
      "... 8 more fields ..."
    ],
    "not_available": "Accuracy metrics and feature importance are not exposed by the ML API."
  },
  "tool_calls": [
    {
      "tool": "get_model_information",
      "status": "ok",
      "label": null,
      "error_code": null,
      "latency_ms": 13.8
    }
  ],
  "error": null,
  "notices": [],
  "provider": {
    "name": "groq",
    "model": "<your GROQ_MODEL>"
  },
  "latency_ms": 122.8
}
```

## 5. Off-topic → no tool call

Message: *What is the capital of France?*

```json
{
  "trace_id": "web-0123456789ab",
  "conversation_id": "d47b0b419a5f49309e775e0c131db9a6",
  "kind": "text",
  "assistant_message": "I can only help with exam-score predictions and the model behind them.",
  "predictions": [],
  "clarification": null,
  "model_information": null,
  "tool_calls": [],
  "error": null,
  "notices": [],
  "provider": {
    "name": "groq",
    "model": "<your GROQ_MODEL>"
  },
  "latency_ms": 89.2
}
```

## 6. ML API is down → error (HTTP 200, `kind: "error"`)

Try it: `docker compose stop ml-api`, ask the question from example 1.

```json
{
  "trace_id": "web-0123456789ab",
  "conversation_id": "3f1c9a52d0e84b7aa1c6e2b94d7f05c8",
  "kind": "error",
  "assistant_message": "I couldn't get a prediction because the prediction service is not reachable right now. No prediction was made. Please try again in a moment.",
  "predictions": [],
  "clarification": null,
  "model_information": null,
  "tool_calls": [
    {
      "tool": "predict_exam_score",
      "status": "error",
      "label": "Student",
      "error_code": "ML_API_UNAVAILABLE",
      "latency_ms": 6.4
    }
  ],
  "error": {
    "code": "ML_API_UNAVAILABLE",
    "message": "The prediction service is not reachable.",
    "retryable": true,
    "details": []
  },
  "notices": [],
  "provider": {
    "name": "groq",
    "model": "<your GROQ_MODEL>"
  },
  "latency_ms": 41.2
}
```

## 7. Invalid request → HTTP 422

```json
{
  "trace_id": "3fcab31f25cb4ff9",
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was not valid.",
    "retryable": null,
    "details": [
      "message: Value error, message must not be empty"
    ]
  }
}
```

---

## Error codes

All non-200 responses use `{"trace_id": "...", "error": {"code", "message", "retryable", "details"}}`.

| HTTP | `error.code` | Meaning | Fix |
|---|---|---|---|
| 422 | `INVALID_REQUEST` | Empty / too long message, bad `conversation_id` | Fix the request |
| 413 | `REQUEST_TOO_LARGE` | Body over 8192 bytes | Send less |
| 503 | `LLM_NOT_CONFIGURED` | No API key / model name set | Edit `.env`, restart orchestrator |
| 502 | `LLM_AUTH_FAILED` | Provider rejected the key | Check `GROQ_API_KEY` |
| 502 | `LLM_MODEL_NOT_FOUND` | Provider does not know the model | Check `GROQ_MODEL` / `ollama pull` |
| 503 | `LLM_RATE_LIMITED` | Provider rate limit | Wait and retry |
| 502 | `LLM_TOOLS_UNSUPPORTED` | The model cannot do tool calling | Choose a tool-capable model |
| 504 | `LLM_TIMEOUT` | Provider too slow (local CPU?) | Smaller model / raise `LLM_TIMEOUT_SECONDS` |
| 502 | `LLM_UNAVAILABLE` | Provider not reachable (`ollama serve`?) | Start Ollama / check network |
| 502 | `LLM_BAD_RESPONSE` | Provider replied with something unusable | Retry; try another model |
| 502 | `LLM_INVALID_TOOL_ARGS`, `LLM_INVALID_TOOL_CALL`, `LLM_TOO_MANY_CALLS`, `LLM_TOO_MANY_STEPS` | The LLM misbehaved with tools | Rephrase; use a stronger model |
| 502 | `MCP_UNAVAILABLE` | Orchestrator cannot reach the MCP server | Start it / check `MCP_SERVER_URL` |

**Inside a `200` response** with `kind: "error"`: `ML_API_UNAVAILABLE`, `ML_API_TIMEOUT`, `ML_API_REJECTED_INPUT`,
`ML_API_MODEL_NOT_LOADED`, `ML_API_ERROR`, `ML_API_BAD_RESPONSE`, `INVALID_INPUT`.

## Health endpoints

`GET /health`
```json
{"status":"ok","service":"orchestrator","provider":{"name":"groq","model":"<your GROQ_MODEL>"},"llm_configured":true,"llm_note":null}
```

`GET /health/dependencies` (example with a stand-in LLM)
```json
{"llm":{"ok":true},"provider":"ollama","model":"fake-model","mcp":{"ok":true,"tools":["predict_exam_score","get_model_information"]},"ml_api":{"ok":true,"model_version":"1792163631093186959","error":null},"all_ok":true}
```
