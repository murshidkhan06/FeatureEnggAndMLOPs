import { FEATURE_LABELS, formatScore, labelFor } from "../featureLabels";
import type { PredictionResult } from "../types";

/** Everything on this card comes straight from the ML API's response - never from the LLM's text. */
export function PredictionCard({ prediction }: { prediction: PredictionResult }) {
  const entries = Object.entries(prediction.features);
  return (
    <section className="card" aria-label={`Model output for ${prediction.label}`}>
      <header>
        <span className="tag model">Model output</span>
        <h3>{prediction.label}</h3>
      </header>
      <p className="score" aria-label="Predicted final score">
        <span className="score-value">{formatScore(prediction.predicted_final_score)}</span>
        <span className="score-label">predicted final score</span>
      </p>
      <details open>
        <summary>Input features used</summary>
        <table>
          <tbody>
            {entries.map(([key, value]) => (
              <tr key={key}>
                <th scope="row">{labelFor(key)}</th>
                <td>
                  {String(value)} {FEATURE_LABELS[key]?.unit && <span className="unit">{FEATURE_LABELS[key].unit}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <p className="meta">
        {prediction.model_version && <>Model version: <code>{prediction.model_version}</code> </>}
        {prediction.ml_request_id && <>· ML request: <code>{prediction.ml_request_id}</code></>}
      </p>
    </section>
  );
}
