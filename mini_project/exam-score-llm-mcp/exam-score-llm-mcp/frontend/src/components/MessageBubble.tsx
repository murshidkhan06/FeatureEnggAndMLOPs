import { formatScore } from "../featureLabels";
import type { ChatMessage } from "../types";
import { ClarificationNotice } from "./ClarificationNotice";
import { ErrorBanner } from "./ErrorBanner";
import { PredictionCard } from "./PredictionCard";

export function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return <div className="bubble user">{message.text}</div>;
  }
  if (message.role === "problem") {
    return <ErrorBanner title={message.title} detail={message.detail} code={message.code} traceId={message.traceId} />;
  }

  const r = message.response;
  // A difference is arithmetic on two REAL outputs; it says nothing about cause and effect.
  const showDifference = r.predictions.length === 2;
  const diff = showDifference ? r.predictions[1].predicted_final_score - r.predictions[0].predicted_final_score : 0;

  return (
    <div className={`bubble assistant kind-${r.kind}`}>
      {r.kind === "error" && r.error ? (
        <div role="alert" className="error-inline">
          <strong>The prediction could not be completed</strong>
          <p>{r.error.message}</p>
          {r.error.details.length > 0 && <ul>{r.error.details.map((d) => <li key={d}>{d}</li>)}</ul>}
          <p className="meta">Code: <code>{r.error.code}</code>{r.error.retryable ? " · You can try again." : ""}</p>
        </div>
      ) : (
        <>
          <span className="tag llm">Assistant explanation</span>
          <p className="assistant-text">{r.assistant_message}</p>
        </>
      )}

      {r.clarification && r.clarification.length > 0 && <ClarificationNotice items={r.clarification} />}

      {r.predictions.length > 0 && (
        <div className="cards">
          {r.predictions.map((p) => <PredictionCard key={p.label + (p.ml_request_id ?? "")} prediction={p} />)}
        </div>
      )}

      {showDifference && (
        <p className="diff">
          Difference between the two model outputs: <strong>{diff > 0 ? "+" : ""}{formatScore(Math.round(diff * 10) / 10)}</strong> points.
          This compares two predictions; it does not prove that the changed input <em>caused</em> the change.
        </p>
      )}

      {r.model_information && (
        <details className="info">
          <summary>Model information (from the ML API)</summary>
          <pre>{JSON.stringify(r.model_information, null, 2)}</pre>
        </details>
      )}

      {r.notices.map((n) => <p key={n} className="notice" role="note">⚠ {n}</p>)}

      <p className="meta footnote">
        Trace ID: <code>{r.trace_id}</code> · {Math.round(r.latency_ms)} ms
        {r.tool_calls.length > 0 && <> · tools: {r.tool_calls.map((t) => `${t.tool}(${t.status})`).join(", ")}</>}
      </p>
    </div>
  );
}
