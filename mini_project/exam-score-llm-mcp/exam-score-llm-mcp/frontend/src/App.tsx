import { useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { Composer } from "./components/Composer";
import { ProviderBadge } from "./components/ProviderBadge";
import { SamplePanel } from "./components/SamplePanel";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, loading, provider, llmNote, backendDown, send, reset } = useChat();
  const [draft, setDraft] = useState("");

  const submit = async () => {
    const text = draft;
    setDraft("");
    await send(text);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Exam Score Assistant</h1>
          <p className="subtitle">Natural language → LLM → MCP → ML API. The LLM explains; the trained model predicts.</p>
        </div>
        <div className="topbar-actions">
          <ProviderBadge provider={provider} note={llmNote} backendDown={backendDown} />
          <button type="button" className="secondary" onClick={() => { reset(); setDraft(""); }}>New conversation</button>
        </div>
      </header>

      {backendDown && (
        <div className="banner" role="alert">
          Cannot reach the orchestrator. Start it (see README) and then send a message again.
        </div>
      )}
      {!backendDown && llmNote && (
        <div className="banner warn" role="alert">{llmNote}</div>
      )}

      <main className="layout">
        <section className="main-col">
          <ChatWindow messages={messages} loading={loading} />
          <Composer value={draft} onChange={setDraft} onSend={submit} loading={loading} />
        </section>
        <aside className="side-col">
          <SamplePanel onPick={setDraft} disabled={loading} />
          <section className="panel">
            <h2>Who does what?</h2>
            <ul className="roles">
              <li><strong>LLM</strong> understands wording and explains results.</li>
              <li><strong>MCP server</strong> offers safe tools.</li>
              <li><strong>ML model</strong> computes the number.</li>
            </ul>
          </section>
        </aside>
      </main>
    </div>
  );
}
