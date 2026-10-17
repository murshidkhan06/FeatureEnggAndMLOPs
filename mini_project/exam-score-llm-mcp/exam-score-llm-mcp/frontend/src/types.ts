// src/types.ts
// These types mirror orchestrator/app/schemas/api.py by hand. If you change one side, change the other.

export type FeatureValue = string | number;

export interface PredictionResult {
  label: string;
  predicted_final_score: number; // exactly what the ML API returned
  features: Record<string, FeatureValue>; // the validated inputs sent to the model
  model_version: string | null;
  ml_request_id: string | null;
}

export interface ToolCallRecord {
  tool: string;
  status: "ok" | "error" | "rejected";
  label: string | null;
  error_code: string | null;
  latency_ms: number | null;
}

export interface FieldProblem {
  field: string;
  problem: string;
}

export interface ProfileClarification {
  label: string;
  missing_fields: string[];
  problems: FieldProblem[];
  understood_values: Record<string, FeatureValue>;
}

export interface ErrorInfo {
  code: string;
  message: string;
  retryable: boolean | null;
  details: string[];
}

export interface ProviderInfo {
  name: string;
  model: string;
}

export type ResponseKind = "prediction" | "clarification" | "information" | "text" | "error";

export interface ChatResponse {
  trace_id: string;
  conversation_id: string;
  kind: ResponseKind;
  assistant_message: string;
  predictions: PredictionResult[];
  clarification: ProfileClarification[] | null;
  model_information: Record<string, unknown> | null;
  tool_calls: ToolCallRecord[];
  error: ErrorInfo | null;
  notices: string[];
  provider: ProviderInfo;
  latency_ms: number;
}

export interface HealthResponse {
  status: "ok";
  service: string;
  provider: ProviderInfo;
  llm_configured: boolean;
  llm_note: string | null;
}

/** What the UI keeps for each message in the chat history. */
export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; response: ChatResponse }
  | { id: string; role: "problem"; title: string; detail: string; code?: string; traceId?: string };
