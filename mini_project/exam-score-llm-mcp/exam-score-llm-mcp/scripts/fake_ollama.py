"""scripts/fake_ollama.py -- a tiny STAND-IN for Ollama, for offline testing ONLY.

It speaks just enough of Ollama's API (/api/tags and the OpenAI-style /v1/chat/completions) to let you
exercise the whole stack without a GPU, a model download or a Groq key. It is NOT a language model:
it recognises the "complete profile" sample sentence with regular expressions, asks the tool for a
prediction, and then repeats the number the tool returned. Anything else gets a polite fixed reply.

    python scripts/fake_ollama.py --port 11434
    # then in orchestrator/.env:  LLM_PROVIDER=ollama  OLLAMA_MODEL=fake-model

Teaching point: the orchestrator cannot tell this from a real model, which is exactly why every
safety check lives in code and not in the prompt.
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "fake-model"


def _num(pattern: str, text: str):
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else None


def extract_features(text: str) -> dict | None:
    """Pull the 10 fields out of the 'complete profile' sample sentence; None if anything is missing."""
    tests = re.search(r"scored\s+(\d+(?:\.\d+)?),?\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", text, re.I)
    income = re.search(r"\b(Low|Medium|High)\s+income", text, re.I)
    city = re.search(r"lives in\s+([A-Za-z]+)", text, re.I)
    date = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    found = {
        "study_hours": _num(r"studies\s+(\d+(?:\.\d+)?)\s+hours", text),
        "attendance_pct": _num(r"(\d+(?:\.\d+)?)%\s+attendance", text),
        "shoe_size": _num(r"shoe size\s+(\d+(?:\.\d+)?)", text),
        "lucky_number": _num(r"lucky number\s+(\d+)", text),
    }
    if not (tests and income and city and date) or any(v is None for v in found.values()):
        return None
    found.update(mock_test_1=float(tests[1]), mock_test_2=float(tests[2]), mock_test_3=float(tests[3]),
                 income_bracket=income[1].capitalize(), city=city[1].capitalize(), enrollment_date=date[1])
    found["lucky_number"] = int(found["lucky_number"])
    return found


def reply_for(messages: list[dict]) -> dict:
    last = messages[-1]
    if last["role"] == "tool":
        data = json.loads(last["content"]) if last["content"].strip().startswith("{") else {}
        score = data.get("predicted_final_score")
        if score is None:
            return {"content": "The prediction service reported a problem, so I have no score to share."}
        return {"content": f"The model's predicted final score is {score}. (Stand-in assistant: this wording is canned.)"}
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user" and isinstance(m.get("content"), str))
    features = extract_features(user_text)
    if features:
        call = {"id": "call_fake_1", "type": "function",
                "function": {"name": "predict_exam_score", "arguments": json.dumps(features)}}
        return {"content": None, "tool_calls": [call]}
    return {"content": "I am a stand-in model for offline testing. Try the 'Complete profile' sample question."}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            return self._send(200, {"models": [{"name": MODEL}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        message = {"role": "assistant", **reply_for(body.get("messages", []))}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        self._send(200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": MODEL,
                         "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

    def log_message(self, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"fake Ollama listening on http://{args.host}:{args.port}  (model name: {MODEL})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
