"use client";

import { Activity, BarChart3, Bot, Database, FileText, Loader2, Network, Play, Radio, RefreshCw, Send } from "lucide-react";
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

type IntegrationStatus = {
  name: string;
  role: string;
  status: string;
  required_package: string | null;
  configured: boolean;
  package_available: boolean;
  notes: string;
};

type IntegrationReport = {
  mode: string;
  ready_count: number;
  integration_count: number;
  integrations: IntegrationStatus[];
};

type EvalCaseResult = {
  case_id: string;
  query: string;
  passed: boolean;
  grounding_score: number;
  retrieved_sources: number;
  latency_ms: number;
  failed_agents: number;
  missing_expected_terms: string[];
  missing_source_terms: string[];
};

type EvalReport = {
  case_count: number;
  passed_count: number;
  failed_count: number;
  pass_rate: number;
  average_grounding_score: number;
  average_latency_ms: number;
  average_retrieved_sources: number;
  total_failed_agents: number;
  cases: EvalCaseResult[];
};

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [query, setQuery] = useState("Which resume projects mention RAG, FastAPI, Next.js, LangGraph, Qdrant, or Neo4j?");
  const [documentPath, setDocumentPath] = useState("D:/desktop/resume/6/resume-Zhuoying Li.pdf");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [metrics, setMetrics] = useState<HealthMetrics | null>(null);
  const [integrations, setIntegrations] = useState<IntegrationReport | null>(null);
  const [evaluation, setEvaluation] = useState<EvalReport | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [mode, setMode] = useState<"query" | "stream">("query");
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const sourceCount = result?.sources.length ?? 0;
  const displayedAnswer = result?.answer ?? streamedAnswer;
  const groundingScore = useMemo(() => {
    const value = result?.metrics.grounding_score;
    return typeof value === "number" ? value.toFixed(2) : "0.00";
  }, [result]);

  useEffect(() => {
    void refreshDashboard();
  }, []);

  async function refreshDashboard() {
    await Promise.all([refreshMetrics(), refreshIntegrations()]);
  }

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

  async function refreshIntegrations() {
    try {
      const response = await fetch(`${apiBaseUrl}/health/integrations`);
      if (!response.ok) {
        throw new Error(`Integrations request failed with ${response.status}`);
      }
      setIntegrations((await response.json()) as IntegrationReport);
    } catch {
      setIntegrations(null);
    }
  }

  async function runEvaluation() {
    setEvaluating(true);
    setEvaluationError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw new Error(`Evaluation request failed with ${response.status}`);
      }
      setEvaluation((await response.json()) as EvalReport);
    } catch (caught) {
      setEvaluationError(caught instanceof Error ? caught.message : "Evaluation failed");
    } finally {
      setEvaluating(false);
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
      await refreshDashboard();
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
        <button className="iconButton" type="button" onClick={refreshDashboard} aria-label="Refresh dashboard">
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

        <div className="integrationPanel">
          <div className="sectionHeader">
            <Network size={18} />
            <h2>Integrations</h2>
          </div>
          <div className="integrationSummary">
            <span>{integrations?.mode ?? "unavailable"}</span>
            <strong>
              {integrations ? `${integrations.ready_count}/${integrations.integration_count} ready` : "0/0 ready"}
            </strong>
          </div>
          <div className="integrationList">
            {(integrations?.integrations ?? []).map((item) => (
              <article className="integrationItem" key={item.name}>
                <div>
                  <strong>{item.name}</strong>
                  <span>{item.required_package ?? "built-in"}</span>
                </div>
                <span className={`statusPill ${item.status}`}>{item.status}</span>
              </article>
            ))}
          </div>
        </div>

        <div className="evaluationPanel">
          <div className="sectionHeader">
            <BarChart3 size={18} />
            <h2>Evaluation</h2>
          </div>
          <div className="evaluationHeader">
            <div>
              <span>Default regression set</span>
              <strong>{evaluation ? `${evaluation.passed_count}/${evaluation.case_count} passed` : "Not run"}</strong>
            </div>
            <button className="secondaryButton" type="button" onClick={runEvaluation} disabled={evaluating}>
              {evaluating ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
              Run Eval
            </button>
          </div>
          {evaluationError && <div className="compactError">{evaluationError}</div>}
          {evaluation && (
            <>
              <div className="evalMetrics">
                <Metric label="Pass rate" value={`${Math.round(evaluation.pass_rate * 100)}%`} />
                <Metric label="Avg grounding" value={evaluation.average_grounding_score.toFixed(2)} />
                <Metric label="Avg latency" value={`${evaluation.average_latency_ms.toFixed(2)} ms`} />
                <Metric label="Failed agents" value={evaluation.total_failed_agents.toString()} />
              </div>
              <div className="caseList">
                {evaluation.cases.map((item) => (
                  <article className="caseItem" key={item.case_id}>
                    <div>
                      <strong>{item.case_id}</strong>
                      <span>{item.query}</span>
                    </div>
                    <span className={`statusPill ${item.passed ? "ready" : "missing_config"}`}>
                      {item.passed ? "PASS" : "FAIL"}
                    </span>
                  </article>
                ))}
              </div>
            </>
          )}
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
