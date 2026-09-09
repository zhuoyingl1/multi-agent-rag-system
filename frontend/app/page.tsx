"use client";

import {
  Activity,
  BarChart3,
  Bot,
  Database,
  FileText,
  Loader2,
  MessageSquarePlus,
  Network,
  Play,
  Radio,
  RefreshCw,
  Send,
  Upload,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

type Source = {
  citation_id: string;
  chunk_id: string;
  title: string | null;
  chunk_type: string;
  score: number;
  retrieval_type: string;
  highlights: string[];
  text: string;
  source_locator: {
    label: string;
  };
};

type QueryResponse = {
  query: string;
  answer: string;
  sources: Source[];
  metrics: Record<string, string | number | boolean>;
};

type UploadResponse = {
  filename: string;
  document_id: string;
  document_path: string;
  content_type: string | null;
  size_bytes: number;
  status: string;
  duplicate: boolean;
};

type DocumentStatusResponse = {
  document_id: string;
  status: string;
  progress_percentage: number;
  current_stage: string;
  stage_details: string;
};

type ConversationResponse = {
  conversation_id: string;
  title: string;
  document_id: string;
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
const defaultQuery = "";

export default function Home() {
  const [query, setQuery] = useState(defaultQuery);
  const [documentId, setDocumentId] = useState("");
  const [documentName, setDocumentName] = useState("");
  const [conversationId, setConversationId] = useState("");
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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const sourceCount = result?.sources.length ?? 0;
  const displayedAnswer = result?.answer ?? streamedAnswer;
  const answerMode = formatAnswerMode(result?.metrics);
  const answerWarning = formatAnswerWarning(result?.metrics);
  const uploadDisabled = !hydrated || uploading || !selectedFile;
  const runDisabled = !hydrated || loading || !query.trim() || !documentId;
  const evaluationDisabled = !hydrated || evaluating;
  const groundingScore = useMemo(() => {
    const value = result?.metrics.grounding_score;
    return typeof value === "number" ? value.toFixed(2) : "0.00";
  }, [result]);

  useEffect(() => {
    setHydrated(true);
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
        throw new Error(await readErrorMessage(response, `Evaluation request failed with ${response.status}`));
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
      const activeConversationId = await ensureConversation();
      if (mode === "stream") {
        const completed = await runStreamQuery(activeConversationId);
        if (!completed) {
          throw new Error("Stream ended before the final answer event.");
        }
      } else {
        await runStandardQuery(activeConversationId);
      }
      await refreshDashboard();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function updateQuery(value: string) {
    setQuery(value);
  }

  function startNewConversation() {
    setConversationId("");
    setQuery("");
    setResult(null);
    setStreamedAnswer("");
    setEvents([]);
    setError(null);
  }

  async function uploadDocument() {
    if (!selectedFile) {
      return;
    }

    setUploading(true);
    setUploadStatus(null);
    setError(null);
    setDocumentId("");
    setConversationId("");
    setResult(null);
    setStreamedAnswer("");
    try {
      const body = new FormData();
      body.append("file", selectedFile);
      const response = await fetch(`${apiBaseUrl}/documents/upload`, {
        method: "POST",
        body,
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Upload request failed with ${response.status}`));
      }
      const uploaded = (await response.json()) as UploadResponse;
      setDocumentName(uploaded.filename);
      setUploadStatus(`Indexing ${uploaded.filename}`);
      await waitForDocument(uploaded.document_id, uploaded.filename);
      setDocumentId(uploaded.document_id);
      setUploadStatus(`Ready: ${uploaded.filename}`);
    } catch (caught) {
      setUploadStatus(null);
      setError(caught instanceof Error ? caught.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function waitForDocument(documentId: string, filename: string) {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      const response = await fetch(`${apiBaseUrl}/documents/${documentId}`);
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Document status request failed with ${response.status}`));
      }
      const status = (await response.json()) as DocumentStatusResponse;
      if (status.status === "completed") {
        return;
      }
      if (status.status === "failed") {
        throw new Error(status.stage_details || "Document indexing failed");
      }
      setUploadStatus(`Indexing ${filename}: ${status.progress_percentage}%`);
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    throw new Error("Document indexing timed out");
  }

  async function ensureConversation() {
    if (conversationId) {
      return conversationId;
    }
    const response = await fetch(`${apiBaseUrl}/conversations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ document_id: documentId, title: documentName || undefined }),
    });
    if (!response.ok) {
      throw new Error(await readErrorMessage(response, `Conversation request failed with ${response.status}`));
    }
    const conversation = (await response.json()) as ConversationResponse;
    setConversationId(conversation.conversation_id);
    return conversation.conversation_id;
  }

  async function runStandardQuery(activeConversationId: string) {
    const response = await fetch(`${apiBaseUrl}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        document_id: documentId,
        conversation_id: activeConversationId,
        require_llm_answer: true,
      }),
    });
    if (!response.ok) {
      throw new Error(await readErrorMessage(response, `Query request failed with ${response.status}`));
    }
    setResult((await response.json()) as QueryResponse);
  }

  async function runStreamQuery(activeConversationId: string) {
    const response = await fetch(`${apiBaseUrl}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        document_id: documentId,
        conversation_id: activeConversationId,
        require_llm_answer: true,
      }),
    });
    if (!response.ok || !response.body) {
      throw new Error(await readErrorMessage(response, `Stream request failed with ${response.status}`));
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

  async function readErrorMessage(response: Response, fallback: string) {
    try {
      const payload = (await response.json()) as { detail?: string | Array<{ msg?: string; loc?: Array<string | number> }> };
      if (typeof payload.detail === "string") {
        return payload.detail;
      }
      if (Array.isArray(payload.detail)) {
        return payload.detail
          .map((item) => {
            const field = item.loc?.slice(1).join(".") || "request";
            return `${field}: ${item.msg || "Invalid value"}`;
          })
          .join("; ");
      }
      return fallback;
    } catch {
      return fallback;
    }
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
    if (eventName === "error" && data) {
      const payload = JSON.parse(data) as { detail?: string };
      throw new Error(payload.detail || "Streaming query failed");
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

          <label htmlFor="indexedDocument">Indexed document</label>
          <input id="indexedDocument" value={documentName} placeholder="Upload a document to begin" readOnly />

          <label htmlFor="documentUpload">Upload document</label>
          <div className="uploadRow">
            <input
              id="documentUpload"
              type="file"
              accept=".md,.markdown,.txt,.json,.csv,.pdf"
              onChange={(event) => {
                setSelectedFile(event.target.files?.[0] ?? null);
                setUploadStatus(null);
              }}
            />
            <button className="secondaryButton uploadButton" type="button" onClick={uploadDocument} disabled={uploadDisabled}>
              {uploading ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
              Upload
            </button>
          </div>
          {uploadStatus && <div className="uploadStatus">{uploadStatus}</div>}

          <label htmlFor="query">Question</label>
          <textarea id="query" value={query} onChange={(event) => updateQuery(event.target.value)} rows={7} />

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

          <button className="primaryButton" type="submit" disabled={runDisabled}>
            {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Run
          </button>
        </form>

        <section className="answerPanel">
          <div className="sectionHeader answerHeader">
            <div className="answerTitle">
              <Bot size={18} />
              <h2>Answer</h2>
            </div>
            <button
              className="smallIconButton"
              type="button"
              onClick={startNewConversation}
              disabled={!hydrated || loading || !documentId}
              aria-label="Start new conversation"
              title="Start new conversation"
            >
              <MessageSquarePlus size={17} />
            </button>
          </div>

          {error && <div className="errorBox">{error}</div>}
          {!error && !displayedAnswer && <div className="emptyState">Run a query to inspect the grounded answer.</div>}
          {displayedAnswer && (
            <>
              <div className="answerText">{displayedAnswer}</div>
              {answerWarning && <div className="answerWarning">{answerWarning}</div>}
            </>
          )}
        </section>
      </section>

      <section className="inspector">
        <div className="metricStrip">
          <Metric label="Sources" value={sourceCount.toString()} />
          <Metric label="Grounding" value={groundingScore} />
          <Metric label="Answer mode" value={answerMode} />
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
                  <span>[{source.citation_id}]</span>
                  <span>{source.title ?? source.chunk_id}</span>
                  <span>{source.source_locator.label}</span>
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
            {events.length === 0
              ? <span>No stream events yet.</span>
              : events.map((event, index) => <span key={`${event}-${index}`}>{event}</span>)}
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
                <p>{item.notes}</p>
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
            <button className="secondaryButton" type="button" onClick={runEvaluation} disabled={evaluationDisabled}>
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

function formatAnswerMode(metrics: QueryResponse["metrics"] | undefined) {
  const answerType = typeof metrics?.answer_type === "string" ? metrics.answer_type : "";
  const answerModel = typeof metrics?.answer_model === "string" ? metrics.answer_model : "";
  if (!answerType) {
    return "Not run";
  }
  if (answerType === "llm" && answerModel) {
    return answerModel;
  }
  return answerType.replaceAll("_", " ");
}

function formatAnswerWarning(metrics: QueryResponse["metrics"] | undefined) {
  const answerType = typeof metrics?.answer_type === "string" ? metrics.answer_type : "";
  const answerError = typeof metrics?.answer_error === "string" ? metrics.answer_error : "";
  if (answerType !== "deterministic_fallback" || !answerError) {
    return "";
  }
  return `Fallback answer shown because the LLM answer provider was unavailable: ${answerError}`;
}
