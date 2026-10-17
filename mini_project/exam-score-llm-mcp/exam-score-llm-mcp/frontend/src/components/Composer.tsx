import { type FormEvent, type KeyboardEvent } from "react";
import { MAX_MESSAGE_CHARS } from "../hooks/useChat";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  loading: boolean;
}

export function Composer({ value, onChange, onSend, loading }: Props) {
  const trimmed = value.trim();
  const tooLong = value.length > MAX_MESSAGE_CHARS;
  const canSend = !loading && trimmed.length > 0 && !tooLong;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (canSend) onSend();
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // Enter sends, Shift+Enter makes a new line
      if (canSend) onSend();
    }
  };

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="question" className="sr-only">Your question</label>
      <textarea
        id="question"
        rows={3}
        value={value}
        placeholder="Describe a student in plain English, e.g. 'studies 12 hours a week, 88% attendance…'"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKey}
        aria-invalid={tooLong}
      />
      <div className="composer-row">
        <span className={tooLong ? "count over" : "count"}>{value.length}/{MAX_MESSAGE_CHARS}</span>
        <button type="submit" disabled={!canSend}>{loading ? "Working…" : "Send"}</button>
      </div>
      {tooLong && <p className="field-error" role="alert">Message is too long. Please shorten it to {MAX_MESSAGE_CHARS} characters.</p>}
    </form>
  );
}
