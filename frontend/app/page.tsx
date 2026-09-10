"use client";

import {
  Activity,
  BarChart3,
  Bot,
  Check,
  ChevronLeft,
  ChevronRight,
  Database,
  FileSearch,
  FileText,
  FolderTree,
  History,
  Loader2,
  MessageSquarePlus,
  Network,
  Pencil,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Search,
  Send,
  Trash2,
  Upload,
  X,
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
  knowledge_space_id: string | null;
};

type UploadQueueItem = {
  id: string;
  file: File;
  status: "queued" | "uploading" | "processing" | "completed" | "failed";
  progress: number;
  documentId?: string;
  error?: string;
};

type DocumentStatusResponse = {
  document_id: string;
  filename: string;
  document_path: string;
  status: string;
  progress_percentage: number;
  current_stage: string;
  stage_details: string;
  index_version: string | null;
  expected_index_version: string;
  chunking_version: string | null;
  expected_chunking_version: string;
  embedding_model: string | null;
  expected_embedding_model: string;
  indexed_at: string | null;
  chunk_count: number;
  index_stale: boolean;
  knowledge_space_id: string | null;
};

type DocumentListResponse = {
  documents: DocumentStatusResponse[];
  total: number;
  skip: number;
  limit: number;
};

type ChunkPreview = {
  chunk_id: string;
  index: number;
  chunk_type: string;
  text: string;
  source_locator: {
    label: string;
  };
};

type DocumentChunkListResponse = {
  document_id: string;
  filename: string;
  chunks: ChunkPreview[];
  total: number;
  skip: number;
  limit: number;
};

type KnowledgeSpace = {
  knowledge_space_id: string;
  name: string;
  description: string;
  document_count: number;
  created_at: string;
  updated_at: string;
};

type KnowledgeSpaceListResponse = {
  knowledge_spaces: KnowledgeSpace[];
  total: number;
};

type ConversationResponse = {
  conversation_id: string;
  title: string;
  document_id: string | null;
  knowledge_space_id: string | null;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
};

type ConversationMessage = {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  metadata: Record<string, unknown>;
};

type ConversationSummary = Omit<ConversationResponse, "messages"> & {
  message_count: number;
};

type ConversationListResponse = {
  conversations: ConversationSummary[];
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
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationMessages, setConversationMessages] = useState<ConversationMessage[]>([]);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [deletingConversationId, setDeletingConversationId] = useState("");
  const [editingConversationId, setEditingConversationId] = useState("");
  const [conversationTitleDraft, setConversationTitleDraft] = useState("");
  const [updatingConversationId, setUpdatingConversationId] = useState("");
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
  const [uploadQueue, setUploadQueue] = useState<UploadQueueItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [documentStatus, setDocumentStatus] = useState<DocumentStatusResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentStatusResponse[]>([]);
  const [knowledgeSpaces, setKnowledgeSpaces] = useState<KnowledgeSpace[]>([]);
  const [knowledgeSpaceId, setKnowledgeSpaceId] = useState("");
  const [newSpaceName, setNewSpaceName] = useState("");
  const [creatingSpace, setCreatingSpace] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [chunkPreview, setChunkPreview] = useState<DocumentChunkListResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewQuery, setPreviewQuery] = useState("");
  const [previewType, setPreviewType] = useState("");
  const [hydrated, setHydrated] = useState(false);
  const previewLimit = 5;

  const sourceCount = result?.sources.length ?? 0;
  const displayedAnswer = result?.answer ?? streamedAnswer;
  const answerMode = formatAnswerMode(result?.metrics);
  const answerWarning = formatAnswerWarning(result?.metrics);
  const queuedUploadCount = uploadQueue.filter((item) => item.status === "queued").length;
  const uploadDisabled = !hydrated || uploading || queuedUploadCount === 0;
  const scopedDocuments = useMemo(
    () =>
      knowledgeSpaceId
        ? documents.filter((document) => document.knowledge_space_id === knowledgeSpaceId)
        : documents,
    [documents, knowledgeSpaceId],
  );
  const scopeReady = knowledgeSpaceId
    ? scopedDocuments.some((document) => document.status === "completed" && !document.index_stale)
    : Boolean(documentId && documentStatus?.status === "completed" && !documentStatus.index_stale);
  const runDisabled =
    !hydrated ||
    loading ||
    reindexing ||
    deleting ||
    !query.trim() ||
    !scopeReady;
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
    await Promise.all([
      refreshMetrics(),
      refreshIntegrations(),
      refreshDocuments(),
      refreshKnowledgeSpaces(),
      refreshConversations(),
    ]);
  }

  async function refreshConversations(
    selectedDocumentId = documentId,
    selectedKnowledgeSpaceId = knowledgeSpaceId,
  ) {
    if (!selectedDocumentId && !selectedKnowledgeSpaceId) {
      setConversations([]);
      return;
    }
    const params = new URLSearchParams({ limit: "50" });
    if (selectedKnowledgeSpaceId) {
      params.set("knowledge_space_id", selectedKnowledgeSpaceId);
    } else {
      params.set("document_id", selectedDocumentId);
    }
    try {
      const response = await fetch(`${apiBaseUrl}/conversations?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`Conversation list request failed with ${response.status}`);
      }
      const catalog = (await response.json()) as ConversationListResponse;
      setConversations(catalog.conversations);
    } catch {
      setConversations([]);
    }
  }

  async function refreshKnowledgeSpaces() {
    try {
      const response = await fetch(`${apiBaseUrl}/knowledge-spaces?limit=100`);
      if (!response.ok) {
        throw new Error(`Knowledge space list request failed with ${response.status}`);
      }
      const catalog = (await response.json()) as KnowledgeSpaceListResponse;
      setKnowledgeSpaces(catalog.knowledge_spaces);
    } catch {
      setKnowledgeSpaces([]);
    }
  }

  async function refreshDocuments() {
    try {
      const response = await fetch(`${apiBaseUrl}/documents?limit=100`);
      if (!response.ok) {
        throw new Error(`Document list request failed with ${response.status}`);
      }
      const catalog = (await response.json()) as DocumentListResponse;
      setDocuments(catalog.documents);
    } catch {
      setDocuments([]);
    }
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
      await loadConversation(activeConversationId, false);
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
    setConversationMessages([]);
    setQuery("");
    setResult(null);
    setStreamedAnswer("");
    setEvents([]);
    setError(null);
  }

  async function createKnowledgeSpace() {
    const name = newSpaceName.trim();
    if (!name) {
      return;
    }
    setCreatingSpace(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/knowledge-spaces`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Knowledge space request failed with ${response.status}`));
      }
      const created = (await response.json()) as KnowledgeSpace;
      setKnowledgeSpaces((current) => [created, ...current]);
      setKnowledgeSpaceId(created.knowledge_space_id);
      setNewSpaceName("");
      clearSelectedDocument();
      setConversations([]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Knowledge space creation failed");
    } finally {
      setCreatingSpace(false);
    }
  }

  function selectKnowledgeSpace(selectedSpaceId: string) {
    setKnowledgeSpaceId(selectedSpaceId);
    clearSelectedDocument();
    setConversationId("");
    setConversationMessages([]);
    setResult(null);
    setStreamedAnswer("");
    setEvents([]);
    setError(null);
    void refreshConversations("", selectedSpaceId);
  }

  function clearSelectedDocument() {
    setDocumentId("");
    setDocumentName("");
    setDocumentStatus(null);
    setUploadStatus(null);
    setChunkPreview(null);
    setPreviewQuery("");
    setPreviewType("");
  }

  function selectUploadFiles(files: File[]) {
    const seen = new Set<string>();
    const queue = files.flatMap((file) => {
      const signature = `${file.name}-${file.size}-${file.lastModified}`;
      if (seen.has(signature)) {
        return [];
      }
      seen.add(signature);
      return [{ id: signature, file, status: "queued" as const, progress: 0 }];
    });
    setUploadQueue(queue);
    setUploadStatus(null);
  }

  function updateUploadItem(itemId: string, updates: Partial<Omit<UploadQueueItem, "id" | "file">>) {
    setUploadQueue((current) =>
      current.map((item) => (item.id === itemId ? { ...item, ...updates } : item)),
    );
  }

  async function uploadDocuments() {
    const pendingItems = uploadQueue.filter((item) => item.status === "queued");
    if (pendingItems.length === 0) {
      return;
    }

    setUploading(true);
    setUploadStatus(null);
    setError(null);
    setDocumentId("");
    setDocumentStatus(null);
    setConversationId("");
    setConversationMessages([]);
    setResult(null);
    setStreamedAnswer("");
    let completedCount = 0;
    let failedCount = 0;
    let lastCompleted: UploadResponse | null = null;
    for (const item of pendingItems) {
      updateUploadItem(item.id, { status: "uploading", progress: 5, error: undefined });
      setUploadStatus(`Uploading ${item.file.name}`);
      try {
        const body = new FormData();
        body.append("file", item.file);
        if (knowledgeSpaceId) {
          body.append("knowledge_space_id", knowledgeSpaceId);
        }
        const response = await fetch(`${apiBaseUrl}/documents/upload`, {
          method: "POST",
          body,
        });
        if (!response.ok) {
          throw new Error(await readErrorMessage(response, `Upload request failed with ${response.status}`));
        }
        const uploaded = (await response.json()) as UploadResponse;
        updateUploadItem(item.id, { status: "processing", progress: 10, documentId: uploaded.document_id });
        setDocumentId(uploaded.document_id);
        setDocumentName(uploaded.filename);
        await waitForDocument(uploaded.document_id, uploaded.filename, (status) => {
          updateUploadItem(item.id, {
            status: status.status === "completed" ? "completed" : "processing",
            progress: status.progress_percentage,
          });
        });
        updateUploadItem(item.id, { status: "completed", progress: 100 });
        completedCount += 1;
        lastCompleted = uploaded;
      } catch (caught) {
        failedCount += 1;
        updateUploadItem(item.id, {
          status: "failed",
          error: caught instanceof Error ? caught.message : "Upload failed",
        });
      }
    }

    try {
      await Promise.all([refreshDocuments(), refreshKnowledgeSpaces()]);
      if (lastCompleted) {
        await loadChunkPreview(lastCompleted.document_id);
      }
    } finally {
      setUploadStatus(
        failedCount > 0
          ? `Completed ${completedCount} of ${pendingItems.length}; ${failedCount} failed`
          : `Completed ${completedCount} of ${pendingItems.length} documents`,
      );
      setUploading(false);
    }
  }

  async function waitForDocument(
    documentId: string,
    filename: string,
    onProgress?: (status: DocumentStatusResponse) => void,
  ): Promise<DocumentStatusResponse> {
    let status: DocumentStatusResponse;
    try {
      status = await waitForDocumentStream(documentId, filename, onProgress);
    } catch {
      return pollDocumentStatus(documentId, filename, onProgress);
    }
    if (status.status === "failed") {
      throw new Error(status.stage_details || "Document indexing failed");
    }
    return status;
  }

  async function waitForDocumentStream(
    documentId: string,
    filename: string,
    onProgress?: (status: DocumentStatusResponse) => void,
  ): Promise<DocumentStatusResponse> {
    const response = await fetch(`${apiBaseUrl}/documents/${documentId}/progress/stream`);
    if (!response.ok || !response.body) {
      throw new Error(`Document progress stream failed with ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        if (part.startsWith(":")) {
          continue;
        }
        const eventName = part.match(/^event: (.+)$/m)?.[1] ?? "progress";
        const data = part.match(/^data: (.+)$/m)?.[1];
        if (!data) {
          continue;
        }
        const payload = JSON.parse(data) as DocumentStatusResponse | { detail: string };
        if (eventName === "error") {
          throw new Error("detail" in payload ? payload.detail : "Document progress stream failed");
        }
        const status = payload as DocumentStatusResponse;
        updateDocumentProgress(status, filename, onProgress);
        if (eventName === "failed" || status.status === "failed") {
          return status;
        }
        if (eventName === "complete" || status.status === "completed") {
          return status;
        }
      }
    }
    throw new Error("Document progress stream ended before indexing completed");
  }

  async function pollDocumentStatus(
    documentId: string,
    filename: string,
    onProgress?: (status: DocumentStatusResponse) => void,
  ): Promise<DocumentStatusResponse> {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      const response = await fetch(`${apiBaseUrl}/documents/${documentId}/progress`);
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Document status request failed with ${response.status}`));
      }
      const status = (await response.json()) as DocumentStatusResponse;
      updateDocumentProgress(status, filename, onProgress);
      if (status.status === "completed") {
        return status;
      }
      if (status.status === "failed") {
        throw new Error(status.stage_details || "Document indexing failed");
      }
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    throw new Error("Document indexing timed out");
  }

  function updateDocumentProgress(
    status: DocumentStatusResponse,
    filename: string,
    onProgress?: (status: DocumentStatusResponse) => void,
  ) {
    setDocumentStatus(status);
    onProgress?.(status);
    if (status.status === "processing") {
      setUploadStatus(`${status.current_stage || "Indexing"} ${filename}: ${status.progress_percentage}%`);
    }
  }

  async function refreshDocumentIndex() {
    if (!documentId) {
      return;
    }
    setReindexing(true);
    setError(null);
    setConversationId("");
    setConversationMessages([]);
    setResult(null);
    setStreamedAnswer("");
    try {
      const retry = documentStatus?.status === "failed";
      const action = retry ? "retry" : "reindex";
      const response = await fetch(`${apiBaseUrl}/documents/${documentId}/${action}`, { method: "POST" });
      if (!response.ok) {
        const fallback = `${retry ? "Retry" : "Reindex"} request failed with ${response.status}`;
        throw new Error(await readErrorMessage(response, fallback));
      }
      setUploadStatus(`${retry ? "Retrying" : "Reindexing"} ${documentName}`);
      const status = await waitForDocument(documentId, documentName);
      setUploadStatus(`Ready: ${documentName} (${status.chunk_count} chunks)`);
      await Promise.all([refreshDocuments(), refreshKnowledgeSpaces()]);
      await loadChunkPreview(documentId);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Document indexing request failed");
    } finally {
      setReindexing(false);
    }
  }

  function selectDocument(selectedDocumentId: string) {
    const selected = documents.find((document) => document.document_id === selectedDocumentId) ?? null;
    setDocumentId(selected?.document_id ?? "");
    setDocumentName(selected?.filename ?? "");
    setDocumentStatus(selected);
    setConversationId("");
    setConversationMessages([]);
    setResult(null);
    setStreamedAnswer("");
    setEvents([]);
    setError(null);
    setPreviewQuery("");
    setPreviewType("");
    if (!selected) {
      setUploadStatus(null);
      setChunkPreview(null);
    } else if (selected.index_stale) {
      setUploadStatus(`Index update required: ${selected.filename}`);
    } else if (selected.status === "completed") {
      setUploadStatus(`Ready: ${selected.filename} (${selected.chunk_count} chunks)`);
    } else {
      setUploadStatus(`${selected.current_stage}: ${selected.progress_percentage}%`);
    }
    if (selected) {
      void loadChunkPreview(selected.document_id, 0, "", "");
    }
    void refreshConversations(selected?.document_id ?? "", knowledgeSpaceId);
  }

  async function deleteDocument() {
    if (!documentId || !window.confirm(`Delete ${documentName} and all of its indexed data?`)) {
      return;
    }
    setDeleting(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${documentId}`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Delete request failed with ${response.status}`));
      }
      setDocumentId("");
      setDocumentName("");
      setDocumentStatus(null);
      setConversationId("");
      setConversationMessages([]);
      setConversations([]);
      setResult(null);
      setStreamedAnswer("");
      setEvents([]);
      setUploadStatus(null);
      setChunkPreview(null);
      setPreviewQuery("");
      setPreviewType("");
      await Promise.all([refreshDocuments(), refreshKnowledgeSpaces()]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  }

  async function loadChunkPreview(
    selectedDocumentId = documentId,
    skip = 0,
    queryValue = previewQuery,
    typeValue = previewType,
  ) {
    if (!selectedDocumentId) {
      setChunkPreview(null);
      return;
    }
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const params = new URLSearchParams({ skip: skip.toString(), limit: previewLimit.toString() });
      if (queryValue.trim()) {
        params.set("q", queryValue.trim());
      }
      if (typeValue) {
        params.set("chunk_type", typeValue);
      }
      const response = await fetch(`${apiBaseUrl}/documents/${selectedDocumentId}/chunks?${params.toString()}`);
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Chunk preview request failed with ${response.status}`));
      }
      setChunkPreview((await response.json()) as DocumentChunkListResponse);
    } catch (caught) {
      setPreviewError(caught instanceof Error ? caught.message : "Chunk preview failed");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function ensureConversation() {
    if (conversationId) {
      return conversationId;
    }
    const title = buildConversationTitle(query);
    const response = await fetch(`${apiBaseUrl}/conversations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(
        knowledgeSpaceId
          ? {
              knowledge_space_id: knowledgeSpaceId,
              title,
            }
          : { document_id: documentId, title },
      ),
    });
    if (!response.ok) {
      throw new Error(await readErrorMessage(response, `Conversation request failed with ${response.status}`));
    }
    const conversation = (await response.json()) as ConversationResponse;
    setConversationId(conversation.conversation_id);
    return conversation.conversation_id;
  }

  async function loadConversation(selectedConversationId: string, displayLatestAnswer = true) {
    setConversationLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/conversations/${selectedConversationId}`);
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Conversation request failed with ${response.status}`));
      }
      const conversation = (await response.json()) as ConversationResponse;
      setConversationId(conversation.conversation_id);
      setConversationMessages(conversation.messages);
      if (displayLatestAnswer) {
        const latestAnswer = [...conversation.messages].reverse().find((message) => message.role === "assistant");
        setQuery("");
        setResult(null);
        setStreamedAnswer(latestAnswer?.content ?? "");
        setEvents([]);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Conversation loading failed");
    } finally {
      setConversationLoading(false);
    }
  }

  async function deleteConversation(selectedConversationId: string, title: string) {
    if (!window.confirm(`Delete conversation ${title}?`)) {
      return;
    }
    setDeletingConversationId(selectedConversationId);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/conversations/${selectedConversationId}`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Conversation delete failed with ${response.status}`));
      }
      if (conversationId === selectedConversationId) {
        startNewConversation();
      }
      await refreshConversations();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Conversation deletion failed");
    } finally {
      setDeletingConversationId("");
    }
  }

  function editConversationTitle(conversation: ConversationSummary) {
    setEditingConversationId(conversation.conversation_id);
    setConversationTitleDraft(conversation.title);
  }

  async function updateConversationTitle(selectedConversationId: string) {
    const title = conversationTitleDraft.trim();
    if (!title) {
      return;
    }
    setUpdatingConversationId(selectedConversationId);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/conversations/${selectedConversationId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, `Conversation update failed with ${response.status}`));
      }
      const updated = (await response.json()) as ConversationSummary;
      setConversations((current) =>
        current.map((conversation) =>
          conversation.conversation_id === selectedConversationId ? updated : conversation,
        ),
      );
      setEditingConversationId("");
      setConversationTitleDraft("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Conversation update failed");
    } finally {
      setUpdatingConversationId("");
    }
  }

  async function runStandardQuery(activeConversationId: string) {
    const response = await fetch(`${apiBaseUrl}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        ...(knowledgeSpaceId ? { knowledge_space_id: knowledgeSpaceId } : { document_id: documentId }),
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
        ...(knowledgeSpaceId ? { knowledge_space_id: knowledgeSpaceId } : { document_id: documentId }),
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

          <label htmlFor="knowledgeSpace">Knowledge space</label>
          <div className="spaceSelectRow">
            <select
              id="knowledgeSpace"
              value={knowledgeSpaceId}
              onChange={(event) => selectKnowledgeSpace(event.target.value)}
            >
              <option value="">Single document</option>
              {knowledgeSpaces.map((space) => (
                <option key={space.knowledge_space_id} value={space.knowledge_space_id}>
                  {space.name} ({space.document_count})
                </option>
              ))}
            </select>
            <FolderTree size={18} aria-hidden="true" />
          </div>
          <div className="spaceCreateRow">
            <input
              value={newSpaceName}
              onChange={(event) => setNewSpaceName(event.target.value)}
              placeholder="New knowledge space"
              maxLength={120}
              aria-label="New knowledge space name"
            />
            <button
              className="smallIconButton"
              type="button"
              onClick={createKnowledgeSpace}
              disabled={!hydrated || creatingSpace || !newSpaceName.trim()}
              aria-label="Create knowledge space"
              title="Create knowledge space"
            >
              {creatingSpace ? <Loader2 className="spin" size={17} /> : <Plus size={17} />}
            </button>
          </div>

          <label htmlFor="indexedDocument">{knowledgeSpaceId ? "Document to inspect" : "Indexed document"}</label>
          <div className="indexedDocumentRow">
            <select id="indexedDocument" value={documentId} onChange={(event) => selectDocument(event.target.value)}>
              <option value="">{knowledgeSpaceId ? "Select a document in this space" : "Select a document"}</option>
              {scopedDocuments.map((document) => (
                <option key={document.document_id} value={document.document_id}>
                  {document.filename} ({document.status})
                </option>
              ))}
            </select>
            <button
              className="smallIconButton"
              type="button"
              onClick={refreshDocumentIndex}
              disabled={!hydrated || !documentId || uploading || reindexing || loading}
              aria-label={documentStatus?.status === "failed" ? "Retry document indexing" : "Reindex document"}
              title={documentStatus?.status === "failed" ? "Retry document indexing" : "Reindex document"}
            >
              <RefreshCw className={reindexing ? "spin" : ""} size={17} />
            </button>
            <button
              className="smallIconButton dangerIconButton"
              type="button"
              onClick={deleteDocument}
              disabled={!hydrated || !documentId || uploading || reindexing || deleting || loading}
              aria-label="Delete document"
              title="Delete document"
            >
              {deleting ? <Loader2 className="spin" size={17} /> : <Trash2 size={17} />}
            </button>
          </div>

          <label htmlFor="documentUpload">{knowledgeSpaceId ? "Upload to knowledge space" : "Upload document"}</label>
          <div className="uploadRow">
            <input
              id="documentUpload"
              type="file"
              multiple
              accept=".md,.markdown,.txt,.json,.csv,.pdf"
              onChange={(event) => {
                selectUploadFiles(Array.from(event.target.files ?? []));
                event.currentTarget.value = "";
              }}
            />
            <button className="secondaryButton uploadButton" type="button" onClick={uploadDocuments} disabled={uploadDisabled}>
              {uploading ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
              {uploading ? "Uploading" : queuedUploadCount > 0 ? `Upload ${queuedUploadCount}` : "Upload"}
            </button>
          </div>
          {uploadQueue.length > 0 && (
            <div className="uploadQueue" aria-label="Upload queue">
              {uploadQueue.map((item) => (
                <div className="uploadQueueItem" key={item.id}>
                  <div className="uploadQueueHeader">
                    <div>
                      <strong>{item.file.name}</strong>
                      <span>{formatFileSize(item.file.size)}</span>
                    </div>
                    <span className={`queueStatus ${item.status}`}>{item.status}</span>
                    <button
                      className="queueRemoveButton"
                      type="button"
                      onClick={() => setUploadQueue((current) => current.filter((entry) => entry.id !== item.id))}
                      disabled={uploading}
                      aria-label={`Remove ${item.file.name} from upload queue`}
                      title="Remove from queue"
                    >
                      <X size={15} />
                    </button>
                  </div>
                  <div
                    className="progressTrack"
                    role="progressbar"
                    aria-label={`${item.file.name} upload progress`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={item.progress}
                  >
                    <span style={{ width: `${item.progress}%` }} />
                  </div>
                  {item.error && <span className="uploadQueueError">{item.error}</span>}
                </div>
              ))}
            </div>
          )}
          {uploadStatus && <div className="uploadStatus">{uploadStatus}</div>}
          {reindexing && documentStatus?.status === "processing" && (
            <div
              className="progressTrack"
              role="progressbar"
              aria-label="Document indexing progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={documentStatus.progress_percentage}
            >
              <span style={{ width: `${documentStatus.progress_percentage}%` }} />
            </div>
          )}

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
              disabled={!hydrated || loading || (!documentId && !knowledgeSpaceId)}
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

        <section className="conversationPanel">
          <div className="sectionHeader conversationHeader">
            <div className="answerTitle">
              <History size={18} />
              <h2>Conversation History</h2>
            </div>
            <span>{conversations.length} conversations</span>
          </div>
          {conversations.length === 0 ? (
            <div className="compactEmptyState">Completed conversations for the current scope will appear here.</div>
          ) : (
            <div className="conversationLayout">
              <div className="conversationList">
                {conversations.map((conversation) => (
                  <div
                    className={`conversationRow ${conversation.conversation_id === conversationId ? "active" : ""}`}
                    key={conversation.conversation_id}
                  >
                    {editingConversationId === conversation.conversation_id ? (
                      <form
                        className="conversationEditRow"
                        onSubmit={(event) => {
                          event.preventDefault();
                          void updateConversationTitle(conversation.conversation_id);
                        }}
                      >
                        <input
                          autoFocus
                          value={conversationTitleDraft}
                          onChange={(event) => setConversationTitleDraft(event.target.value)}
                          maxLength={120}
                          aria-label="Conversation title"
                        />
                        <button
                          type="submit"
                          disabled={!conversationTitleDraft.trim() || Boolean(updatingConversationId)}
                          aria-label="Save conversation title"
                          title="Save title"
                        >
                          {updatingConversationId ? <Loader2 className="spin" size={15} /> : <Check size={15} />}
                        </button>
                        <button
                          type="button"
                          onClick={() => setEditingConversationId("")}
                          aria-label="Cancel conversation rename"
                          title="Cancel"
                        >
                          <X size={15} />
                        </button>
                      </form>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => void loadConversation(conversation.conversation_id)}
                          disabled={conversationLoading || loading}
                        >
                          <strong>{conversation.title}</strong>
                          <span>{conversation.message_count} messages</span>
                        </button>
                        <button
                          className="conversationActionButton"
                          type="button"
                          onClick={() => editConversationTitle(conversation)}
                          disabled={Boolean(deletingConversationId) || loading}
                          aria-label={`Rename conversation ${conversation.title}`}
                          title="Rename conversation"
                        >
                          <Pencil size={15} />
                        </button>
                        <button
                          className="conversationActionButton danger"
                          type="button"
                          onClick={() => void deleteConversation(conversation.conversation_id, conversation.title)}
                          disabled={Boolean(deletingConversationId) || loading}
                          aria-label={`Delete conversation ${conversation.title}`}
                          title="Delete conversation"
                        >
                          {deletingConversationId === conversation.conversation_id ? (
                            <Loader2 className="spin" size={15} />
                          ) : (
                            <Trash2 size={15} />
                          )}
                        </button>
                      </>
                    )}
                  </div>
                ))}
              </div>
              <div className="conversationTranscript">
                {conversationLoading ? (
                  <div className="compactEmptyState"><Loader2 className="spin" size={18} /></div>
                ) : conversationMessages.length === 0 ? (
                  <div className="compactEmptyState">Select a conversation to review its messages.</div>
                ) : (
                  conversationMessages.map((message) => (
                    <article className={`conversationMessage ${message.role}`} key={message.message_id}>
                      <span>{message.role === "user" ? "Question" : "Answer"}</span>
                      <p>{message.content}</p>
                    </article>
                  ))
                )}
              </div>
            </div>
          )}
        </section>

        <section className="previewPanel">
          <div className="sectionHeader previewHeader">
            <div className="answerTitle">
              <FileSearch size={18} />
              <h2>Indexed Chunks</h2>
            </div>
            <span>{chunkPreview?.total ?? 0} matches</span>
          </div>
          <div className="previewToolbar">
            <input
              value={previewQuery}
              onChange={(event) => setPreviewQuery(event.target.value)}
              placeholder="Search indexed chunks"
              aria-label="Search indexed chunks"
            />
            <select value={previewType} onChange={(event) => setPreviewType(event.target.value)} aria-label="Chunk type">
              <option value="">All types</option>
              <option value="prose">Prose</option>
              <option value="code">Code</option>
              <option value="table">Table</option>
              <option value="formula">Formula</option>
            </select>
            <button
              className="smallIconButton"
              type="button"
              onClick={() => loadChunkPreview(documentId, 0)}
              disabled={!documentId || previewLoading}
              aria-label="Search chunks"
              title="Search chunks"
            >
              {previewLoading ? <Loader2 className="spin" size={17} /> : <Search size={17} />}
            </button>
          </div>
          {previewError && <div className="compactError">{previewError}</div>}
          {!previewError && !previewLoading && documentId && chunkPreview?.chunks.length === 0 && (
            <div className="previewEmpty">No indexed chunks match these filters.</div>
          )}
          <div className="previewList">
            {(chunkPreview?.chunks ?? []).map((chunk) => (
              <article className="previewItem" key={chunk.chunk_id}>
                <div className="previewMeta">
                  <span>#{chunk.index}</span>
                  <span>{chunk.chunk_type}</span>
                  <span>{chunk.source_locator.label}</span>
                </div>
                <p>{chunk.text}</p>
              </article>
            ))}
          </div>
          {chunkPreview && chunkPreview.total > 0 && (
            <div className="previewFooter">
              <span>
                {chunkPreview.skip + 1}-{Math.min(chunkPreview.skip + chunkPreview.chunks.length, chunkPreview.total)} of {chunkPreview.total}
              </span>
              <div>
                <button
                  className="smallIconButton"
                  type="button"
                  onClick={() => loadChunkPreview(documentId, Math.max(0, chunkPreview.skip - previewLimit))}
                  disabled={previewLoading || chunkPreview.skip === 0}
                  aria-label="Previous chunk page"
                  title="Previous page"
                >
                  <ChevronLeft size={17} />
                </button>
                <button
                  className="smallIconButton"
                  type="button"
                  onClick={() => loadChunkPreview(documentId, chunkPreview.skip + previewLimit)}
                  disabled={previewLoading || chunkPreview.skip + chunkPreview.chunks.length >= chunkPreview.total}
                  aria-label="Next chunk page"
                  title="Next page"
                >
                  <ChevronRight size={17} />
                </button>
              </div>
            </div>
          )}
        </section>

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

function formatFileSize(sizeBytes: number) {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

function buildConversationTitle(question: string) {
  const normalized = question.replace(/\s+/g, " ").trim();
  return normalized.length > 72 ? `${normalized.slice(0, 69)}...` : normalized;
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
