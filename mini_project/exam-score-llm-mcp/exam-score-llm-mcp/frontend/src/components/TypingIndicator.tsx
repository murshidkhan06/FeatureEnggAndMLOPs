export function TypingIndicator() {
  return (
    <div className="bubble assistant typing" role="status" aria-live="polite">
      <span className="dot" /><span className="dot" /><span className="dot" />
      <span className="sr-only">The assistant is working…</span>
      <span className="typing-text">Understanding your question and asking the model…</span>
    </div>
  );
}
