"use client";

import { Activity, Bot, Database, FileText, Loader2, Play, Radio, RefreshCw, Send } from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

type Source = {
  chunk_id: string;
  title: string | null;
  chunk_type: string;
  score: number;
  retrieval_type: string;
  highlights: string[];
  text: string;
};

type QueryResponse = {
  query: string;
  answer: string;
  sources: Source[];
  metrics: Record<string, string | number>;
};

type HealthMetrics = {
  run_count: number;
  average_latency_ms: number;
  average_grounding_score: number;
  average_retrieved_sources: number;
  total_failed_agents: number;
};

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [query, setQuery] = useState("Which resume projects mention RAG, FastAPI, Next.js, LangGraph, Qdrant, or Neo4j?");
  const [documentPath, setDocumentPath] = useState("D:/desktop/resume/6/resume-Zhuoying Li.pdf");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [metrics, setMetrics] = useState<HealthMetrics | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [mode, setMode] = useState<"query" | "stream">("query");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sourceCount = result?.sources.length ?? 0;
  const displayedAnswer = result?.answer ?? streamedAnswer;
  const groundingScore = useMemo(() => {
    const value = result?.metrics.grounding_score;
    return typeof value === "number" ? value.toFixed(2) : "0.00";
  }, [result]);

  useEffect(() => {
    void refreshMetrics();
  }, []);

  async function refreshMetrics() {
    try {
      const response = await fetch(`${apiBaseUrl}/health/metrics`);
      if (!response.ok) {
        throw new Error(`Metrics request failed with ${response.status}`);
      }
      setMetrics((await response.json()) as HealthMetrics);
    } catch {
      setMetrics(null);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setEvents([]);
    setResult(null);
    setStreamedAnswer("");

    try {
      if (mode === "stream") {
        const completed = await runStreamQuery();
        if (!completed) {
          throw new Error("Stream ended before the final answer event.");
        }
      } else {
        await runStandardQuery();
      }
      await refreshMetrics();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function runStandardQuery() {
    const response = await fetch(`${apiBaseUrl}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, document_path: documentPath }),
    });
    if (!response.ok) {
      throw new Error(`Query request failed with ${response.status}`);
    }
    setResult((await response.json()) as QueryResponse);
  }

  async function runStreamQuery() {
    const response = await fetch(`${apiBaseUrl}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, document_path: documentPath }),
    });
    if (!response.ok || !response.body) {
      throw new Error(`Stream request failed with ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let completed = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          completed = handleStreamEvent(part) || completed;
        }
      }
    } catch (caught) {
      if (!completed) {
        throw caught;
      }
    }
    return completed;
  }

  function handleStreamEvent(rawEvent: string) {
    const eventName = rawEvent.match(/^event: (.+)$/m)?.[1] ?? "message";
    const data = rawEvent.match(/^data: (.+)$/m)?.[1];
    setEvents((current) => [...current, eventName]);
    if (eventName === "final" && data) {
      const finalPayload = JSON.parse(data) as QueryResponse;
      setResult(finalPayload);
      setStreamedAnswer(finalPayload.answer);
      return true;
    }
    if (eventName === "answer_delta" && data) {
      const payload = JSON.parse(data) as { delta: string };
      setStreamedAnswer((current) => current + payload.delta);
    }
    return false;
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Local RAG Console</p>
          <h1>Multi-Agent Research Workflow</h1>
        </div>
        <button className="iconButton" type="button" onClick={refreshMetrics} aria-label="Refresh metrics">
          <RefreshCw size={18} />
        </button>
      </section>

      <section className="workspace">
        <form className="queryPanel" onSubmit={handleSubmit}>
          <div className="sectionHeader">
            <FileText size={18} />
            <h2>Document Query</h2>
          </div>

          <label htmlFor="documentPath">Document path</label>
          <input id="documentPath" value={documentPath} onChange={(event) => setDocumentPath(event.target.value)} />

          <label htmlFor="query">Question</label>
          <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} rows={7} />

          <div className="modeSwitch" aria-label="Query mode">
            <button type="button" className={mode === "query" ? "active" : ""} onClick={() => setMode("query")}>
              <Send size={16} />
              Query
            </button>
            <button type="button" className={mode === "stream" ? "active" : ""} onClick={() => setMode("stream")}>
              <Radio size={16} />
              Stream
            </button>
          </div>

          <button className="primaryButton" type="submit" disabled={loading || !query.trim() || !documentPath.trim()}>
            {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Run
          </button>
        </form>

        <section className="answerPanel">
          <div className="sectionHeader">
            <Bot size={18} />
            <h2>Answer</h2>
          </div>

          {error && <div className="errorBox">{error}</div>}
          {!error && !displayedAnswer && <div className="emptyState">Run a query to inspect the grounded answer.</div>}
          {displayedAnswer && <pre className="answerText">{displayedAnswer}</pre>}
        </section>
      </section>

      <section className="inspector">
        <div className="metricStrip">
          <Metric label="Sources" value={sourceCount.toString()} />
          <Metric label="Grounding" value={groundingScore} />
          <Metric label="Runs" value={metrics?.run_count.toString() ?? "0"} />
          <Metric label="Avg latency" value={`${metrics?.average_latency_ms.toFixed(2) ?? "0.00"} ms`} />
        </div>

        <div className="sourcePanel">
          <div className="sectionHeader">
            <Database size={18} />
            <h2>Sources</h2>
          </div>
          <div className="sourceList">
            {(result?.sources ?? []).map((source) => (
              <article className="sourceItem" key={source.chunk_id}>
                <div className="sourceMeta">
                  <span>{source.title ?? source.chunk_id}</span>
                  <span>{source.retrieval_type}</span>
                  <span>score {source.score}</span>
                </div>
                <p>{source.text}</p>
                <div className="chips">
                  {source.highlights.map((highlight) => (
                    <span key={highlight}>{highlight}</span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>

        <div className="eventPanel">
          <div className="sectionHeader">
            <Activity size={18} />
            <h2>Events</h2>
          </div>
          <div className="events">
            {events.length === 0 ? <span>No stream events yet.</span> : events.map((event, index) => <span key={`${event}-${index}`}>{event}</span>)}
          </div>
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
