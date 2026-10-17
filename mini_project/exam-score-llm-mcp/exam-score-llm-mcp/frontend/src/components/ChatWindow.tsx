import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";

export function ChatWindow({ messages, loading }: { messages: ChatMessage[]; loading: boolean }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" }); // optional chaining: jsdom has no scrollIntoView
  }, [messages, loading]);

  return (
    <div className="chat" role="log" aria-live="polite" aria-label="Conversation">
      {messages.length === 0 && !loading && (
        <div className="empty">
          <p><strong>Ask about a student's exam score.</strong></p>
          <p>The language model reads your sentence and asks the trained model for the number. If something is missing, it will ask you instead of guessing.</p>
        </div>
      )}
      {messages.map((m) => <MessageBubble key={m.id} message={m} />)}
      {loading && <TypingIndicator />}
      <div ref={endRef} />
    </div>
  );
}
