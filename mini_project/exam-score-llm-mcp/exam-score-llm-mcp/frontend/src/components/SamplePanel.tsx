import { SAMPLES } from "../samples";

interface Props {
  onPick: (text: string) => void;
  disabled?: boolean;
}

/** Clicking a sample only fills the input box. The prediction is still computed by the real model. */
export function SamplePanel({ onPick, disabled }: Props) {
  return (
    <section className="panel" aria-label="Sample questions">
      <h2>Try a sample question</h2>
      <ul>
        {SAMPLES.map((s) => (
          <li key={s.title}>
            <button type="button" className="sample" disabled={disabled} onClick={() => onPick(s.text)}>
              <strong>{s.title}</strong>
              <span>{s.what}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
