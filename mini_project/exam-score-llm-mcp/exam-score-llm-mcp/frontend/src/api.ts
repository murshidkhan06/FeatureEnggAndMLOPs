// src/api.ts -- the ONLY place the browser talks to the backend.
// The browser calls the ORCHESTRATOR. It never calls Groq, Ollama, MCP or the ML API directly,
// so no API key ever exists in the browser.
import type { ChatResponse, HealthResponse } from "./types";

export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8002").replace(/\/$/, "");

/** A failure we can show to the user. */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly traceId?: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** A short id sent as X-Request-ID; it comes back as `trace_id` and appears in every service's logs. */
export function newTraceId(): string {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  return "web-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

const CHAT_TIMEOUT_MS = 120_000; // local models on a CPU can be slow

async function parseError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    if (body?.error?.message) {
      return new ApiError(body.error.message, body.error.code ?? "ERROR", body.trace_id, response.status);
    }
  } catch {
    /* not JSON -- fall through */
  }
  return new ApiError(`The server answered with HTTP ${response.status}.`, "HTTP_ERROR", undefined, response.status);
}

export async function postChat(message: string, conversationId: string | null, signal?: AbortSignal): Promise<ChatResponse> {
  const timeout = AbortSignal.timeout(CHAT_TIMEOUT_MS);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Request-ID": newTraceId() },
      body: JSON.stringify({ message, ...(conversationId ? { conversation_id: conversationId } : {}) }),
      signal: combined,
    });
  } catch (err) {
    if (signal?.aborted) throw err; // the user pressed Reset: not an error to display
    if (timeout.aborted) throw new ApiError("The request took too long and was stopped.", "CLIENT_TIMEOUT");
    throw new ApiError(`Cannot reach the orchestrator at ${API_BASE_URL}. Is it running?`, "NETWORK_ERROR");
  }
  if (!response.ok) throw await parseError(response);
  return (await response.json()) as ChatResponse;
}

export async function deleteConversation(conversationId: string): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
  } catch {
    /* best effort: the server forgets idle conversations anyway */
  }
}

export async function getHealth(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
  } catch {
    throw new ApiError(`Cannot reach the orchestrator at ${API_BASE_URL}.`, "NETWORK_ERROR");
  }
  if (!response.ok) throw await parseError(response);
  return (await response.json()) as HealthResponse;
}
