import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { MAX_MESSAGE_CHARS } from "../hooks/useChat";
import { SAMPLES } from "../samples";
import { FEATURES, chatResponse, mockFetch, prediction } from "./fixtures";

afterEach(() => vi.unstubAllGlobals());

const box = () => screen.getByLabelText("Your question") as HTMLTextAreaElement;
const sendButton = () => screen.getByRole("button", { name: /send|working/i });

async function ask(text: string) {
  const user = userEvent.setup();
  await user.click(box());
  await user.paste(text); // paste: fast and avoids {} / [] special characters in user-event's keyboard syntax
  await user.click(sendButton());
  return user;
}

describe("provider badge", () => {
  it("shows provider and model but nothing secret", async () => {
    mockFetch({});
    render(<App />);
    expect(await screen.findByText(/LLM: groq · test-model/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/api[_-]?key|gsk_|bearer/i);
  });

  it("warns when the LLM is not configured", async () => {
    const { json } = mockFetch({});
    mockFetch({ health: () => json({ status: "ok", service: "orchestrator", provider: { name: "groq", model: "(not set)" }, llm_configured: false, llm_note: "GROQ_API_KEY is not set." }) });
    render(<App />);
    expect(await screen.findByText("GROQ_API_KEY is not set.", { selector: ".banner" })).toBeInTheDocument();
  });

  it("shows an offline banner when the orchestrator cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    render(<App />);
    expect(await screen.findByText("Backend offline")).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the orchestrator/)).toBeInTheDocument();
  });
});

describe("sending messages", () => {
  it("disables Send for empty / whitespace input", async () => {
    mockFetch({});
    render(<App />);
    const user = userEvent.setup();
    expect(sendButton()).toBeDisabled();
    await user.type(box(), "   ");
    expect(sendButton()).toBeDisabled();
    await user.type(box(), "hi");
    expect(sendButton()).toBeEnabled();
  });

  it("blocks messages over the length limit and says why", async () => {
    mockFetch({});
    render(<App />);
    const user = userEvent.setup();
    await user.click(box());
    await user.paste("x".repeat(MAX_MESSAGE_CHARS + 1));
    expect(sendButton()).toBeDisabled();
    expect(screen.getByRole("alert", { name: "" })).toHaveTextContent(/too long/i);
  });

  it("shows a loading indicator while waiting, then the answer; sends X-Request-ID", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const m = mockFetch({
      chat: async () => { await gate; return m.json(chatResponse({ assistant_message: "Hi there" })); },
    });
    render(<App />);
    await ask("hello");
    expect(await screen.findByText(/asking the model/i)).toBeInTheDocument();
    expect(sendButton()).toBeDisabled();
    release();
    expect(await screen.findByText("Hi there")).toBeInTheDocument();
    expect(screen.queryByText(/asking the model/i)).not.toBeInTheDocument();

    const chatCall = m.calls.find((c) => c.url.endsWith("/chat"))!;
    expect((chatCall.init!.headers as Record<string, string>)["X-Request-ID"]).toMatch(/^web-[0-9a-f]{12}$/);
    expect(JSON.parse(chatCall.init!.body as string)).toEqual({ message: "hello" });
  });

  it("only talks to the orchestrator - never to Groq, Ollama, MCP or the ML API", async () => {
    const m = mockFetch({});
    render(<App />);
    await ask("hello");
    await screen.findByText("Hello.");
    for (const c of m.calls) expect(c.url).toMatch(/^http:\/\/localhost:8002\//);
  });

  it("reuses the conversation id on the next message", async () => {
    const m = mockFetch({});
    render(<App />);
    const user = await ask("first");
    await screen.findAllByText("Hello.");
    await user.click(box());
    await user.paste("second");
    await user.click(sendButton());
    await waitFor(() => expect(m.calls.filter((c) => c.url.endsWith("/chat"))).toHaveLength(2));
    const bodies = m.calls.filter((c) => c.url.endsWith("/chat")).map((c) => JSON.parse(c.init!.body as string));
    expect(bodies[0].conversation_id).toBeUndefined();
    expect(bodies[1].conversation_id).toBe("conv-12345678");
  });

  it("Enter sends and Shift+Enter does not", async () => {
    const m = mockFetch({});
    render(<App />);
    const user = userEvent.setup();
    await user.click(box());
    await user.type(box(), "line one{Shift>}{Enter}{/Shift}line two");
    expect(m.calls.filter((c) => c.url.endsWith("/chat"))).toHaveLength(0);
    expect(box().value).toContain("\n");
    await user.keyboard("{Enter}");
    await waitFor(() => expect(m.calls.filter((c) => c.url.endsWith("/chat"))).toHaveLength(1));
  });
});

describe("rendering model output", () => {
  it("shows the exact score and the exact input features from the response", async () => {
    mockFetch({ chat: (_u, _i) => new Response(JSON.stringify(chatResponse({
      kind: "prediction", assistant_message: "The model predicts a final score of 95.2.", predictions: [prediction(95.2)],
    })), { status: 200 }) });
    render(<App />);
    await ask("predict");
    const card = await screen.findByRole("region", { name: "Model output for Student" });
    expect(within(card).getByLabelText("Predicted final score")).toHaveTextContent("95.2");
    for (const [key, label] of [["study_hours", "Study hours"], ["city", "City"], ["enrollment_date", "Enrollment date"]] as const) {
      const row = within(card).getByText(label).closest("tr")!;
      expect(row).toHaveTextContent(String(FEATURES[key]));
    }
    expect(within(card).getByText("Model output")).toBeInTheDocument();
    expect(screen.getByText("Assistant explanation")).toBeInTheDocument();
  });

  it("formats whole-number scores with one decimal and does not round other values", async () => {
    mockFetch({ chat: () => new Response(JSON.stringify(chatResponse({ kind: "prediction", predictions: [prediction(80), prediction(81.25, "Other")] })), { status: 200 }) });
    render(<App />);
    await ask("x");
    const cards = await screen.findAllByLabelText("Predicted final score");
    expect(cards[0]).toHaveTextContent("80.0");
    expect(cards[1]).toHaveTextContent("81.25");
  });

  it("compares exactly two predictions without claiming causation", async () => {
    mockFetch({ chat: () => new Response(JSON.stringify(chatResponse({
      kind: "prediction", predictions: [prediction(70.5, "Student A"), prediction(78.0, "Student B")],
    })), { status: 200 }) });
    render(<App />);
    await ask("compare");
    expect(await screen.findByText(/Difference between the two model outputs/)).toHaveTextContent("+7.5");
    expect(screen.getByText(/does not prove/i)).toBeInTheDocument();
  });

  it("does not show a difference for a single prediction", async () => {
    mockFetch({ chat: () => new Response(JSON.stringify(chatResponse({ kind: "prediction", predictions: [prediction(70)] })), { status: 200 }) });
    render(<App />);
    await ask("one");
    await screen.findByLabelText("Predicted final score");
    expect(screen.queryByText(/Difference between/)).not.toBeInTheDocument();
  });

  it("shows a clarification with NO prediction card", async () => {
    mockFetch({ chat: () => new Response(JSON.stringify(chatResponse({
      kind: "clarification", assistant_message: "I need a few more details.",
      clarification: [{ label: "Student", missing_fields: ["mock_test_1", "city"], problems: [{ field: "study_hours", problem: "you described hours per DAY" }], understood_values: { attendance_pct: 85 } }],
    })), { status: 200 }) });
    render(<App />);
    await ask("6 hours daily");
    expect(await screen.findByText(/No prediction yet/)).toBeInTheDocument();
    expect(screen.getByText("Mock test 1")).toBeInTheDocument();
    expect(screen.getByText(/hours per DAY/)).toBeInTheDocument();
    expect(screen.getByText(/Attendance = 85/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Predicted final score")).not.toBeInTheDocument();
  });

  it("shows an ML failure (kind=error) as an alert, with no score", async () => {
    mockFetch({ chat: () => new Response(JSON.stringify(chatResponse({
      kind: "error", assistant_message: "", error: { code: "ML_API_UNAVAILABLE", message: "The prediction service is not reachable.", retryable: true, details: [] },
    })), { status: 200 }) });
    render(<App />);
    await ask("predict");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The prediction service is not reachable.");
    expect(alert).toHaveTextContent("ML_API_UNAVAILABLE");
    expect(screen.queryByLabelText("Predicted final score")).not.toBeInTheDocument();
  });

  it("shows notices and the trace id", async () => {
    mockFetch({ chat: () => new Response(JSON.stringify(chatResponse({ notices: ["The explanation was replaced because it quoted a score the model did not produce."] })), { status: 200 }) });
    render(<App />);
    await ask("x");
    expect(await screen.findByRole("note")).toHaveTextContent(/replaced/);
    expect(screen.getByText("trace-abc12345")).toBeInTheDocument();
  });
});

describe("errors from the orchestrator", () => {
  it("shows the server's error message, code and trace id", async () => {
    mockFetch({ chat: () => new Response(JSON.stringify({ trace_id: "tr-failed-01", error: { code: "LLM_RATE_LIMITED", message: "The language model is busy. Try again soon.", retryable: true, details: [] } }), { status: 503 }) });
    render(<App />);
    await ask("hi");
    const alert = await screen.findByRole("alert", {}, { timeout: 2000 });
    expect(alert).toHaveTextContent("The language model is busy");
    expect(alert).toHaveTextContent("LLM_RATE_LIMITED");
    expect(alert).toHaveTextContent("tr-failed-01");
    // loading has ended: the button says "Send" again (it stays disabled only because the box is empty)
    expect(sendButton()).toHaveTextContent("Send");
    expect(screen.queryByText(/asking the model/i)).not.toBeInTheDocument();
  });

  it("handles a non-JSON error body", async () => {
    mockFetch({ chat: () => new Response("<html>bad gateway</html>", { status: 502 }) });
    render(<App />);
    await ask("hi");
    expect(await screen.findByText(/HTTP 502/)).toBeInTheDocument();
  });

  it("handles a network failure", async () => {
    const m = mockFetch({ chat: () => { throw new TypeError("Failed to fetch"); } });
    render(<App />);
    await ask("hi");
    expect(await screen.findByText(/Cannot reach the orchestrator at http:\/\/localhost:8002/)).toBeInTheDocument();
    expect(m.fn).toHaveBeenCalled();
  });
});

describe("samples and reset", () => {
  it("clicking a sample fills the box but does not send", async () => {
    const m = mockFetch({});
    render(<App />);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /Missing details/ }));
    expect(box().value).toBe(SAMPLES[1].text);
    expect(m.calls.filter((c) => c.url.endsWith("/chat"))).toHaveLength(0);
  });

  it("every sample fits within the message limit", () => {
    for (const s of SAMPLES) expect(s.text.length).toBeLessThanOrEqual(MAX_MESSAGE_CHARS);
  });

  it("'New conversation' clears the chat, deletes the server conversation and starts a fresh one", async () => {
    const m = mockFetch({});
    render(<App />);
    const user = await ask("first");
    await screen.findByText("Hello.");
    await user.click(screen.getByRole("button", { name: "New conversation" }));
    expect(screen.queryByText("Hello.")).not.toBeInTheDocument();
    expect(screen.getByText(/Ask about a student/)).toBeInTheDocument();
    expect(box().value).toBe("");
    await waitFor(() => expect(m.calls.some((c) => c.init?.method === "DELETE" && c.url.endsWith("/chat/conv-12345678"))).toBe(true));

    await user.click(box());
    await user.paste("again");
    await user.click(sendButton());
    await waitFor(() => expect(m.calls.filter((c) => c.url.endsWith("/chat"))).toHaveLength(2));
    const second = JSON.parse(m.calls.filter((c) => c.url.endsWith("/chat"))[1].init!.body as string);
    expect(second.conversation_id).toBeUndefined();
  });

  it("resetting while a request is running discards its late answer", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const m = mockFetch({ chat: async (_u, init) => {
      await gate;
      if (init?.signal?.aborted) throw new DOMException("aborted", "AbortError");
      return m.json(chatResponse({ assistant_message: "LATE ANSWER" }));
    } });
    render(<App />);
    const user = await ask("slow");
    await screen.findByText(/asking the model/i);
    await user.click(screen.getByRole("button", { name: "New conversation" }));
    await act(async () => { release(); await new Promise((r) => setTimeout(r, 50)); });
    expect(screen.queryByText("LATE ANSWER")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("an old aborted request cannot switch off the loading state of a newer one", async () => {
    let n = 0;
    let releaseFirst!: () => void;
    const first = new Promise<void>((r) => (releaseFirst = r));
    let releaseSecond!: () => void;
    const second = new Promise<void>((r) => (releaseSecond = r));
    const m = mockFetch({ chat: async (_u, init) => {
      n += 1;
      if (n === 1) { await first; throw new DOMException("aborted", "AbortError"); }
      await second;
      return m.json(chatResponse({ assistant_message: "SECOND ANSWER" }));
      void init;
    } });
    render(<App />);
    const user = await ask("one");
    await user.click(screen.getByRole("button", { name: "New conversation" }));
    await user.click(box());
    await user.paste("two");
    await user.click(sendButton());
    await screen.findByText(/asking the model/i);
    await act(async () => { releaseFirst(); await new Promise((r) => setTimeout(r, 30)); });
    expect(screen.getByText(/asking the model/i)).toBeInTheDocument(); // still loading the 2nd request
    await act(async () => { releaseSecond(); });
    expect(await screen.findByText("SECOND ANSWER")).toBeInTheDocument();
  });
});
