import { labelFor } from "../featureLabels";
import type { ProfileClarification } from "../types";

/** Shown when the model was NOT run because something is missing or unclear. */
export function ClarificationNotice({ items }: { items: ProfileClarification[] }) {
  return (
    <div className="clarify" aria-label="Details needed">
      <span className="tag ask">No prediction yet — more information needed</span>
      {items.map((item) => (
        <div key={item.label} className="clarify-item">
          <h4>{item.label}</h4>
          {item.missing_fields.length > 0 && (
            <p>
              Missing: {item.missing_fields.map((f) => <code key={f}>{labelFor(f)}</code>)}
            </p>
          )}
          {item.problems.length > 0 && (
            <ul>
              {item.problems.map((p) => (
                <li key={p.field + p.problem}><strong>{labelFor(p.field)}:</strong> {p.problem}</li>
              ))}
            </ul>
          )}
          {Object.keys(item.understood_values).length > 0 && (
            <p className="meta">
              Understood so far:{" "}
              {Object.entries(item.understood_values).map(([k, v]) => (
                <code key={k}>{labelFor(k)} = {String(v)}</code>
              ))}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
