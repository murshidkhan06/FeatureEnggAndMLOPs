#!/usr/bin/env bash
# scripts/smoke_test.sh -- one-shot health check of the whole stack (works for Compose and for individual terminals).
#   bash scripts/smoke_test.sh
# Override addresses if needed:  ML=http://localhost:8000 MCP=http://localhost:8001 ORCH=http://localhost:8002 UI=http://localhost:5173
set -u
ML=${ML:-http://localhost:8000}; MCP=${MCP:-http://localhost:8001}; ORCH=${ORCH:-http://localhost:8002}; UI=${UI:-http://localhost:5173}
fail=0
check() { # name url
  if out=$(curl -fsS -m 8 "$2" 2>&1); then echo "  OK    $1  $2"; else echo "  FAIL  $1  $2  ($out)"; fail=1; fi
}
echo "1) Is each service up?"
check "ML API         " "$ML/health"
check "MCP server     " "$MCP/health"
check "Orchestrator   " "$ORCH/health"
check "Frontend       " "$UI/"
echo; echo "2) Orchestrator -> LLM, MCP and ML API (deep check)"
curl -sS -m 15 "$ORCH/health/dependencies" | python3 -m json.tool || fail=1
echo; echo "3) A real question (needs a configured LLM)"
resp=$(curl -sS -m 120 "$ORCH/chat" -H "Content-Type: application/json" -d '{"message":"Predict the exam score for a student who studies 12.5 hours per week, has 88% attendance, scored 72, 75 and 70 in the three mock tests, belongs to the Medium income bracket, lives in Pune, enrolled on 2025-06-01, has shoe size 9 and lucky number 42."}')
echo "$resp" | python3 -c '
import json, sys
d = json.load(sys.stdin)
if "error" in d and "kind" not in d:
    print("  The chat request failed:", d["error"]["code"], "-", d["error"]["message"]); sys.exit(2)
print("  kind:", d["kind"], "| provider:", d["provider"]["name"], d["provider"]["model"], "| trace:", d["trace_id"])
for p in d["predictions"]: print("  model output:", p["label"], "->", p["predicted_final_score"])
print("  assistant says:", d["assistant_message"][:200])
' || fail=1
echo; [ $fail -eq 0 ] && echo "SMOKE TEST PASSED" || { echo "SMOKE TEST FAILED - see docs/TROUBLESHOOTING.md"; exit 1; }
