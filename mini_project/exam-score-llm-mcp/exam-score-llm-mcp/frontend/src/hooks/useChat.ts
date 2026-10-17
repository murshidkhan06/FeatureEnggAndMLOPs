// src/hooks/useChat.ts -- all chat state in one place, so components stay simple.
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, deleteConversation, getHealth, postChat } from "../api";
import type { ChatMessage, ProviderInfo } from "../types";

export const MAX_MESSAGE_CHARS = 1000; // keep in sync with orchestrator MAX_MESSAGE_CHARS

let counter = 0;
const nextId = () => `m${Date.now().toString(36)}${(counter++).toString(36)}`;

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [provider, setProvider] = useState<ProviderInfo | null>(null);
  const [llmNote, setLlmNote] = useState<string | null>(null);
  const [backendDown, setBackendDown] = useState(false);

  // Refs hold the "current" values for async callbacks without stale closures.
  const conversationRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);

  const refreshHealth = useCallback(async () => {
    try {
      const h = await getHealth();
      setProvider(h.provider);
      setLlmNote(h.llm_configured ? null : (h.llm_note ?? "The language model is not configured."));
      setBackendDown(false);
    } catch {
      setBackendDown(true);
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const send = useCallback(async (raw: string) => {
    const text = raw.trim();
    if (!text || text.length > MAX_MESSAGE_CHARS || busyRef.current) return false;
    busyRef.current = true;
    setLoading(true);
    setMessages((m) => [...m, { id: nextId(), role: "user", text }]);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await postChat(text, conversationRef.current, controller.signal);
      conversationRef.current = response.conversation_id;
      setConversationId(response.conversation_id);
      setProvider(response.provider);
      setBackendDown(false);
      setMessages((m) => [...m, { id: nextId(), role: "assistant", response }]);
    } catch (err) {
      if (controller.signal.aborted) return true; // user pressed Reset: say nothing
      const e = err instanceof ApiError ? err : new ApiError("Something unexpected went wrong.", "UNKNOWN");
      if (e.code === "NETWORK_ERROR") setBackendDown(true);
      setMessages((m) => [
        ...m,
        { id: nextId(), role: "problem", title: "Could not get an answer", detail: e.message, code: e.code, traceId: e.traceId },
      ]);
    } finally {
      // Only the request that is still "current" may clear the busy flag. After Reset, an older
      // request must not switch off the loading state of a newer one.
      if (abortRef.current === controller) {
        abortRef.current = null;
        busyRef.current = false;
        setLoading(false);
      }
    }
    return true;
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    busyRef.current = false;
    const old = conversationRef.current;
    conversationRef.current = null;
    setConversationId(null);
    setMessages([]);
    setLoading(false);
    if (old) void deleteConversation(old);
  }, []);

  return { messages, loading, conversationId, provider, llmNote, backendDown, send, reset, refreshHealth };
}
