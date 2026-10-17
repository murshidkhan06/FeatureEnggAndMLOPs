import type { ChatResponse, PredictionResult } from "../types";

export const FEATURES = {
  study_hours: 12.5, attendance_pct: 88, mock_test_1: 72, mock_test_2: 75, mock_test_3: 70,
  income_bracket: "Medium", city: "Pune", enrollment_date: "2025-06-01", shoe_size: 9, lucky_number: 42,
};

export function prediction(score: number, label = "Student", features: Record<string, string | number> = FEATURES): PredictionResult {
  return { label, predicted_final_score: score, features, model_version: "v-test", ml_request_id: "ml-req-1" };
}

export function chatResponse(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    trace_id: "trace-abc12345", conversation_id: "conv-12345678", kind: "text", assistant_message: "Hello.",
    predictions: [], clarification: null, model_information: null, tool_calls: [], error: null, notices: [],
    provider: { name: "groq", model: "test-model" }, latency_ms: 123, ...overrides,
  };
}

export const HEALTH_OK = { status: "ok", service: "orchestrator", provider: { name: "groq", model: "test-model" }, llm_configured: true, llm_note: null };

type Handler = (url: string, init?: RequestInit) => Response | Promise<Response>;

/** Replace window.fetch with a router. Returns the recorded calls so tests can inspect them. */
export function mockFetch(routes: { health?: Handler; chat?: Handler; del?: Handler }) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const method = init?.method ?? "GET";
    if (url.endsWith("/health")) return routes.health ? routes.health(url, init) : json(HEALTH_OK);
    if (url.endsWith("/chat") && method === "POST") return routes.chat ? routes.chat(url, init) : json(chatResponse());
    if (url.includes("/chat/") && method === "DELETE") return routes.del ? routes.del(url, init) : new Response(null, { status: 204 });
    throw new Error(`unexpected fetch ${method} ${url}`);
  });
  vi.stubGlobal("fetch", fn);
  return { calls, json, fn };
}
