"""FastAPI application for the local multi-agent RAG prototype."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from hashlib import sha256
import json
import os
from queue import Queue
import re
from pathlib import Path
from threading import Thread
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from multi_agent_rag.citations import build_citation_diagnostics, build_source_locator, citation_id, source_locator
from multi_agent_rag.conversation import contextualize_retrieval_query, recent_history
from multi_agent_rag.documents import SUPPORTED_EXTENSIONS, load_document
from multi_agent_rag.evaluation import EvalReport, run_evaluation
from multi_agent_rag.integrations import check_integrations
from multi_agent_rag.ingestion import (
    CHUNKING_VERSION,
    INDEX_VERSION,
    DocumentBusyError,
    DocumentCleanupError,
    DocumentIngestionService,
    document_index_is_stale,
)
from multi_agent_rag.models import AgentPlan, AgentResult, ChunkType, JudgeResult, SearchResult, WorkflowResult
from multi_agent_rag.observability import metrics_registry
from multi_agent_rag.orchestration import create_workflow
from multi_agent_rag.persistence import (
    ChunkRepository,
    ConversationRecord,
    ConversationRepository,
    DocumentRecord,
    DocumentRepository,
    DocumentStatus,
    MongoStore,
)
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.factory import create_document_retriever, create_retriever
from multi_agent_rag.retrieval.embeddings import OllamaEmbeddingService
from multi_agent_rag.retrieval.neo4j_adapter import Neo4jGraphAdapter
from multi_agent_rag.retrieval.vector_index import QdrantDocumentIndex

DEFAULT_DOCUMENT_PATH = Path("examples/sample_docs.md")
DEFAULT_EVAL_CASES_PATH = Path("examples/eval_cases.json")
UPLOAD_DIR = Path("output/uploads")
MONGO_STORE = MongoStore()


class QueryRequest(BaseModel):
    """API request for local document question answering."""

    query: str = Field(min_length=1)
    document_id: str | None = None
    conversation_id: str | None = None
    document_path: str = Field(default=str(DEFAULT_DOCUMENT_PATH), min_length=1)
    orchestrator: str = Field(default="auto", pattern="^(auto|local|langgraph)$")
    retrieval_backend: str = Field(default="qdrant", pattern="^(local|qdrant)$")
    require_llm_answer: bool = True


class EvaluationRequest(BaseModel):
    """API request for deterministic local evaluation."""

    document_path: str = Field(default=str(DEFAULT_DOCUMENT_PATH), min_length=1)
    cases_path: str = Field(default=str(DEFAULT_EVAL_CASES_PATH), min_length=1)
    orchestrator: str = Field(default="auto", pattern="^(auto|local|langgraph)$")
    retrieval_backend: str = Field(default="qdrant", pattern="^(local|qdrant)$")


class UploadResponse(BaseModel):
    """API response for uploaded local documents."""

    filename: str
    document_id: str
    document_path: str
    content_type: str | None
    size_bytes: int
    status: str
    duplicate: bool


class DocumentStatusResponse(BaseModel):
    """Current state of a persistently registered document."""

    document_id: str
    filename: str
    document_path: str
    status: str
    progress_percentage: int
    current_stage: str
    stage_details: str
    index_version: str | None
    expected_index_version: str
    chunking_version: str | None
    expected_chunking_version: str
    embedding_model: str | None
    expected_embedding_model: str
    indexed_at: datetime | None
    chunk_count: int
    index_stale: bool


class DocumentListResponse(BaseModel):
    """Paginated persistent document catalog."""

    documents: list[DocumentStatusResponse]
    total: int
    skip: int
    limit: int


class DocumentDeleteResponse(BaseModel):
    """Confirmation that a document and its indexed data were removed."""

    document_id: str
    filename: str
    deleted: bool


class ChunkPreviewResponse(BaseModel):
    """One stored chunk exposed for document inspection."""

    chunk_id: str
    index: int
    chunk_type: str
    text: str
    source_locator: dict[str, int | str]


class DocumentChunkListResponse(BaseModel):
    """Paginated stored chunk preview for one document."""

    document_id: str
    filename: str
    chunks: list[ChunkPreviewResponse]
    total: int
    skip: int
    limit: int


class ConversationCreateRequest(BaseModel):
    """Create a conversation scoped to one indexed document."""

    document_id: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=120)


def build_app() -> FastAPI:
    app = FastAPI(title="Multi-Agent RAG System V2", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "mode": "deterministic_local",
            "default_document": str(DEFAULT_DOCUMENT_PATH),
            "default_eval_cases": str(DEFAULT_EVAL_CASES_PATH),
        }

    @app.get("/health/metrics")
    def health_metrics() -> dict[str, Any]:
        return metrics_registry.snapshot()

    @app.get("/health/integrations")
    def health_integrations() -> dict[str, Any]:
        return check_integrations(probe_services=True).to_dict()

    @app.post("/documents/upload")
    async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
        service = create_document_ingestion_service()
        return (await save_uploaded_document(file, background_tasks, service)).model_dump()

    @app.get("/documents")
    def list_documents(
        skip: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        status: DocumentStatus | None = Query(default=None),
    ) -> dict[str, Any]:
        repository = DocumentRepository.from_store(MONGO_STORE)
        documents = repository.list(skip=skip, limit=limit, status=status)
        return DocumentListResponse(
            documents=[document_status_payload(document) for document in documents],
            total=repository.count(status),
            skip=skip,
            limit=limit,
        ).model_dump()

    @app.get("/documents/{document_id}")
    def document_status(document_id: str) -> dict[str, Any]:
        try:
            document = DocumentRepository.from_store(MONGO_STORE).get(document_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if document is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        return document_status_payload(document).model_dump()

    @app.get("/documents/{document_id}/chunks")
    def document_chunks(
        document_id: str,
        skip: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
        chunk_type: ChunkType | None = Query(default=None),
        q: str | None = Query(default=None, max_length=200),
    ) -> dict[str, Any]:
        try:
            document = DocumentRepository.from_store(MONGO_STORE).get(document_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if document is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        records, total = ChunkRepository.from_store(MONGO_STORE).list_page(
            document_id,
            skip=skip,
            limit=limit,
            chunk_type=chunk_type.value if chunk_type is not None else None,
            query=q,
        )
        return DocumentChunkListResponse(
            document_id=document_id,
            filename=document.title,
            chunks=[
                ChunkPreviewResponse(
                    chunk_id=record.chunk_id,
                    index=record.index,
                    chunk_type=record.chunk_type,
                    text=record.text,
                    source_locator=build_source_locator(record.index, record.metadata),
                )
                for record in records
            ],
            total=total,
            skip=skip,
            limit=limit,
        ).model_dump()

    @app.post("/documents/{document_id}/reindex")
    def reindex_document(document_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
        service = create_document_ingestion_service()
        try:
            document = service.prepare_reindex(document_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DocumentBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        background_tasks.add_task(service.process, document.document_id, document.file_path)
        return document_status_payload(document).model_dump()

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, Any]:
        service = create_document_ingestion_service()
        try:
            document = service.delete(document_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        except DocumentBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DocumentCleanupError as exc:
            raise HTTPException(status_code=503, detail=f"Document cleanup failed: {exc}") from exc
        return DocumentDeleteResponse(
            document_id=document.document_id,
            filename=document.title,
            deleted=True,
        ).model_dump()

    @app.post("/conversations")
    def create_conversation(request: ConversationCreateRequest) -> dict[str, Any]:
        document = get_ready_document(request.document_id)
        title = request.title or document.title
        conversation = ConversationRepository.from_store(MONGO_STORE).create(
            title=title,
            document_id=document.document_id,
        )
        return conversation_payload(conversation)

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = ConversationRepository.from_store(MONGO_STORE).get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
        return conversation_payload(conversation)

    @app.post("/evaluate")
    def evaluate(request: EvaluationRequest) -> dict[str, Any]:
        return safe_run_evaluation(
            Path(request.document_path),
            Path(request.cases_path),
            request.orchestrator,
            request.retrieval_backend,
        ).to_dict()

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        result = run_query_request(request)
        metrics_registry.record_run(result.metrics)
        return workflow_payload(result)

    @app.post("/query/stream")
    def query_stream(request: QueryRequest) -> StreamingResponse:
        validate_query_source(request)

        def events():
            event_queue: Queue[Any] = Queue()
            completed = object()
            started = perf_counter()
            first_delta_ms: float | None = None

            def on_stage(event: str, value: object) -> None:
                event_queue.put((event, stream_stage_payload(event, value)))

            def on_answer_delta(delta: str) -> None:
                nonlocal first_delta_ms
                if first_delta_ms is None:
                    first_delta_ms = round((perf_counter() - started) * 1000, 2)
                event_queue.put(("answer_delta", {"delta": delta}))

            def run_workflow() -> None:
                try:
                    result = run_query_request(request, on_stage, on_answer_delta)
                    result.metrics["streaming_mode"] = "token" if result.metrics["answer_type"] == "llm" else "complete"
                    result.metrics["time_to_first_token_ms"] = (
                        first_delta_ms if first_delta_ms is not None else result.metrics["latency_ms"]
                    )
                    metrics_registry.record_run(result.metrics)
                    event_queue.put(("final", workflow_payload(result)))
                except HTTPException as exc:
                    event_queue.put(("error", {"detail": str(exc.detail), "status_code": exc.status_code}))
                except Exception as exc:
                    event_queue.put(("error", {"detail": str(exc), "status_code": 500}))
                finally:
                    event_queue.put(completed)

            Thread(target=run_workflow, daemon=True).start()
            while True:
                item = event_queue.get()
                if item is completed:
                    break
                event, payload = item
                yield _sse(event, payload)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run_query_request(
    request: QueryRequest,
    on_stage: Callable[[str, object], None] | None = None,
    on_answer_delta: Callable[[str], None] | None = None,
) -> WorkflowResult:
    history, retrieval_query, conversations = conversation_context(request)
    if request.document_id:
        result = safe_run_document_query(
            request.query,
            request.document_id,
            request.orchestrator,
            request.require_llm_answer,
            on_stage,
            on_answer_delta,
            retrieval_query,
            history,
        )
    else:
        result = safe_run_query(
            request.query,
            Path(request.document_path),
            request.orchestrator,
            request.retrieval_backend,
            request.require_llm_answer,
            on_stage,
            on_answer_delta,
            retrieval_query,
            history,
        )
    if conversations is not None and request.conversation_id is not None:
        conversations.add_turn(
            request.conversation_id,
            user_content=request.query,
            assistant_content=result.answer,
            assistant_metadata={
                "answer_type": result.metrics.get("answer_type", "unknown"),
                "citation_status": result.metrics.get("citation_status", "unknown"),
            },
        )
    return result


def run_query(
    query: str,
    document_path: Path,
    orchestrator: str | None = None,
    retrieval_backend: str | None = None,
    require_llm_answer: bool = False,
    on_stage: Callable[[str, object], None] | None = None,
    on_answer_delta: Callable[[str], None] | None = None,
    retrieval_query: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> WorkflowResult:
    document = load_document(document_path)
    retriever = create_retriever(retrieval_backend)
    retriever.index(chunk_document(document))
    try:
        return create_workflow(
            retriever,
            orchestrator=orchestrator,
            require_llm_answer=require_llm_answer,
            default_answer_provider="ollama" if require_llm_answer else None,
        ).run(query, on_stage, on_answer_delta, retrieval_query, conversation_history)
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()


def run_document_query(
    query: str,
    document_id: str,
    orchestrator: str | None = None,
    require_llm_answer: bool = False,
    on_stage: Callable[[str, object], None] | None = None,
    on_answer_delta: Callable[[str], None] | None = None,
    retrieval_query: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> WorkflowResult:
    document = DocumentRepository.from_store(MONGO_STORE).get(document_id)
    if document is None:
        raise ValueError(f"Document not found: {document_id}")
    if document.status is not DocumentStatus.COMPLETED:
        raise RuntimeError(f"Document is not ready for queries: {document.status.value}")
    if document_index_is_stale(document, configured_embedding_model()):
        raise RuntimeError("Document index is stale. Reindex the document before querying it.")

    retriever = create_document_retriever(document_id)
    try:
        return create_workflow(
            retriever,
            orchestrator=orchestrator,
            require_llm_answer=require_llm_answer,
            default_answer_provider="ollama" if require_llm_answer else None,
        ).run(query, on_stage, on_answer_delta, retrieval_query, conversation_history)
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()


def create_document_ingestion_service() -> DocumentIngestionService:
    embedder = OllamaEmbeddingService(
        base_url=os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434",
        model_name=os.getenv("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text",
        timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS") or "60"),
    )
    vector_index = QdrantDocumentIndex(
        url=os.getenv("QDRANT_URL") or "http://localhost:6333",
        collection=os.getenv("QDRANT_DOCUMENT_COLLECTION") or "document_chunks",
        embedder=embedder,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE") or "50"),
    )
    graph_index = Neo4jGraphAdapter(
        uri=os.getenv("NEO4J_URI") or "bolt://localhost:7687",
        user=os.getenv("NEO4J_USER") or "neo4j",
        password=os.getenv("NEO4J_PASSWORD") or "password123",
        database=os.getenv("NEO4J_DATABASE") or "neo4j",
    )
    return DocumentIngestionService(
        DocumentRepository.from_store(MONGO_STORE),
        ChunkRepository.from_store(MONGO_STORE),
        vector_index,
        graph_index,
        ConversationRepository.from_store(MONGO_STORE),
    )


async def save_uploaded_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    service: DocumentIngestionService,
) -> UploadResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported document extension '{extension}'. Supported extensions: {supported}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).stem).strip(".-") or "document"
    saved_path = UPLOAD_DIR / f"{safe_stem}-{uuid4().hex[:8]}{extension}"
    saved_path.write_bytes(content)
    registered = service.register(
        title=filename,
        file_type=extension.lstrip("."),
        file_path=str(saved_path),
        file_size=len(content),
        file_hash=sha256(content).hexdigest(),
        metadata={"content_type": file.content_type},
    )
    document = registered.document
    if registered.duplicate:
        saved_path.unlink(missing_ok=True)
    else:
        background_tasks.add_task(service.process, document.document_id, document.file_path)
    return UploadResponse(
        filename=document.title,
        document_id=document.document_id,
        document_path=document.file_path,
        content_type=str(document.metadata.get("content_type") or "") or None,
        size_bytes=document.file_size,
        status=document.status.value,
        duplicate=registered.duplicate,
    )


def document_status_payload(document: DocumentRecord) -> DocumentStatusResponse:
    expected_embedding_model = configured_embedding_model()
    return DocumentStatusResponse(
        document_id=document.document_id,
        filename=document.title,
        document_path=document.file_path,
        status=document.status.value,
        progress_percentage=document.progress_percentage,
        current_stage=document.current_stage,
        stage_details=document.stage_details,
        index_version=document.index_version,
        expected_index_version=INDEX_VERSION,
        chunking_version=document.chunking_version,
        expected_chunking_version=CHUNKING_VERSION,
        embedding_model=document.embedding_model,
        expected_embedding_model=expected_embedding_model,
        indexed_at=document.indexed_at,
        chunk_count=document.chunk_count,
        index_stale=document_index_is_stale(document, expected_embedding_model),
    )


def configured_embedding_model() -> str:
    return os.getenv("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text"


def get_ready_document(document_id: str) -> DocumentRecord:
    try:
        document = DocumentRepository.from_store(MONGO_STORE).get(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    if document.status is not DocumentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Document is not ready for queries: {document.status.value}")
    if document_index_is_stale(document, configured_embedding_model()):
        raise HTTPException(status_code=409, detail="Document index is stale. Reindex the document before querying it.")
    return document


def conversation_context(
    request: QueryRequest,
) -> tuple[list[dict[str, str]], str, ConversationRepository | None]:
    if request.conversation_id is None:
        return [], request.query, None
    if request.document_id is None:
        raise HTTPException(status_code=400, detail="A conversation query requires document_id.")

    repository = ConversationRepository.from_store(MONGO_STORE)
    conversation = repository.get(request.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {request.conversation_id}")
    if conversation.document_id != request.document_id:
        raise HTTPException(status_code=400, detail="Conversation does not belong to the requested document.")

    history = recent_history(conversation.messages)
    retrieval_query, _ = contextualize_retrieval_query(request.query, history)
    return history, retrieval_query, repository


def conversation_payload(conversation: ConversationRecord) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "title": conversation.title,
        "document_id": conversation.document_id,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "messages": [
            {
                "message_id": message.message_id,
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata,
            }
            for message in conversation.messages
        ],
    }


def safe_run_query(
    query: str,
    document_path: Path,
    orchestrator: str | None = None,
    retrieval_backend: str | None = None,
    require_llm_answer: bool = False,
    on_stage: Callable[[str, object], None] | None = None,
    on_answer_delta: Callable[[str], None] | None = None,
    retrieval_query: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> WorkflowResult:
    try:
        return run_query(
            query,
            document_path,
            orchestrator=orchestrator,
            retrieval_backend=retrieval_backend,
            require_llm_answer=require_llm_answer,
            on_stage=on_stage,
            on_answer_delta=on_answer_delta,
            retrieval_query=retrieval_query,
            conversation_history=conversation_history,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def safe_run_document_query(
    query: str,
    document_id: str,
    orchestrator: str | None = None,
    require_llm_answer: bool = False,
    on_stage: Callable[[str, object], None] | None = None,
    on_answer_delta: Callable[[str], None] | None = None,
    retrieval_query: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> WorkflowResult:
    try:
        return run_document_query(
            query,
            document_id,
            orchestrator=orchestrator,
            require_llm_answer=require_llm_answer,
            on_stage=on_stage,
            on_answer_delta=on_answer_delta,
            retrieval_query=retrieval_query,
            conversation_history=conversation_history,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def safe_run_evaluation(
    document_path: Path,
    cases_path: Path,
    orchestrator: str | None = None,
    retrieval_backend: str | None = None,
) -> EvalReport:
    if not cases_path.exists():
        raise HTTPException(status_code=400, detail=f"Evaluation cases not found: {cases_path}")
    try:
        return run_evaluation(document_path, cases_path, orchestrator=orchestrator, retrieval_backend=retrieval_backend)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def workflow_payload(result: WorkflowResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "answer": result.answer,
        "workflow_trace": workflow_trace_payload(result),
        "plan": result.plan.__dict__,
        "agents": [agent_payload(agent) for agent in result.agents],
        "grounding": result.grounding.__dict__,
        "citations": build_citation_diagnostics(result.answer, result.sources),
        "sources": [source_payload(source, citation_id(index)) for index, source in enumerate(result.sources)],
        "metrics": result.metrics,
    }


def workflow_trace_payload(result: WorkflowResult) -> dict[str, Any]:
    return {
        "mode": result.metrics.get("mode", "unknown"),
        "evidence_status": result.metrics.get("evidence_status", "unknown"),
        "selected_agents": result.plan.selected_agents,
        "retrieved_sources": result.metrics.get("retrieved_sources", len(result.sources)),
        "candidate_sources": result.metrics.get("candidate_sources", len(result.sources)),
        "reranker": result.metrics.get("reranker", "none"),
        "query_intent": result.metrics.get("query_intent", "general"),
        "query_variants": result.metrics.get("query_variants", 1),
        "selected_k": result.metrics.get("selected_k", len(result.sources)),
        "context_tokens": result.metrics.get("context_tokens", 0),
        "selection_reason": result.metrics.get("selection_reason", "fixed"),
        "conversation_messages": result.metrics.get("conversation_messages", 0),
        "retrieval_query_contextualized": result.metrics.get("retrieval_query_contextualized", False),
        "citation_status": result.metrics.get("citation_status", "no_evidence"),
        "citation_coverage": result.metrics.get("citation_coverage", 0.0),
        "cited_sources": result.metrics.get("cited_sources", 0),
        "invalid_citations": result.metrics.get("invalid_citations", 0),
        "answer_type": result.metrics.get("answer_type", "deterministic"),
        "answer_model": result.metrics.get("answer_model", "template"),
        "answer_error": result.metrics.get("answer_error", ""),
        "completed_agents": result.metrics.get("completed_agents", len(result.agents)),
        "failed_agents": result.metrics.get("failed_agents", 0),
    }


def agent_payload(agent: AgentResult) -> dict[str, Any]:
    return {
        "agent_name": agent.agent_name,
        "task": agent.task,
        "content": agent.content,
        "confidence": agent.confidence,
        "sources": [source_payload(source) for source in agent.sources],
        "error": agent.error,
    }


def source_payload(source: SearchResult, evidence_id: str | None = None) -> dict[str, Any]:
    payload = {
        "chunk_id": source.chunk.chunk_id,
        "document_id": source.chunk.document_id,
        "title": source.chunk.metadata.get("title"),
        "chunk_type": source.chunk.chunk_type.value,
        "score": source.score,
        "retrieval_type": source.retrieval_type.value,
        "highlights": source.highlights,
        "text": source.chunk.text,
        "source_locator": source_locator(source),
    }
    if evidence_id is not None:
        payload["citation_id"] = evidence_id
    return payload


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def validate_query_source(request: QueryRequest) -> None:
    if request.document_id:
        return

    if not Path(request.document_path).is_file():
        raise HTTPException(status_code=400, detail=f"Document not found: {request.document_path}")


def stream_stage_payload(event: str, value: object) -> dict[str, Any]:
    if event == "planning":
        plan = cast(AgentPlan, value)
        return {"selected_agents": plan.selected_agents, "tasks": plan.tasks}
    if event == "retrieval":
        sources = cast(list[SearchResult], value)
        return {
            "count": len(sources),
            "sources": [source_payload(source, citation_id(index)) for index, source in enumerate(sources)],
        }
    if event == "agents":
        agents = cast(list[AgentResult], value)
        return {"agents": [agent_payload(agent) for agent in agents]}
    if event == "judge":
        return cast(JudgeResult, value).__dict__
    raise ValueError(f"Unsupported workflow stream event: {event}")


app = build_app()
