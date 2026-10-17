import type { ProviderInfo } from "../types";

interface Props {
  provider: ProviderInfo | null;
  note: string | null;
  backendDown: boolean;
}

/** Shows which LLM is configured. The orchestrator never sends keys, so there is nothing secret to leak here. */
export function ProviderBadge({ provider, note, backendDown }: Props) {
  if (backendDown) return <span className="badge bad" title="The orchestrator is not reachable">Backend offline</span>;
  if (!provider) return <span className="badge muted">Checking…</span>;
  const cls = note ? "badge warn" : "badge ok";
  return (
    <span className={cls} title={note ?? "Language model used to understand questions and explain results"}>
      LLM: {provider.name} · {provider.model}
    </span>
  );
}
