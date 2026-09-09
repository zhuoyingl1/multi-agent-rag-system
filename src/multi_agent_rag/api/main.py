"""FastAPI application for the local multi-agent RAG prototype."""

from __future__ import annotations

from hashlib import sha256
import json
import os
import re
from uuid import uuid4
from pathlib import Path
from time import sleep
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from multi_agent_rag.documents import load_document
from multi_agent_rag.documents import SUPPORTED_EXTENSIONS
from multi_agent_rag.evaluation import EvalReport, run_evaluation
from multi_agent_rag.integrations import check_integrations
from multi_agent_rag.ingestion import DocumentIngestionService
from multi_agent_rag.models import AgentResult, SearchResult, WorkflowResult
from multi_agent_rag.observability import metrics_registry
from multi_agent_rag.orchestration import create_workflow
from multi_agent_rag.persistence import ChunkRepository, DocumentRecord, DocumentRepository, DocumentStatus, MongoStore
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.factory import create_document_retriever, create_retriever
from multi_agent_rag.retrieval.embeddings import OllamaEmbeddingService
from multi_agent_rag.retrieval.neo4j_adapter import Neo4jGraphAdapter
from multi_agent_rag.retrieval.vector_index import QdrantDocumentIndex

DEFAULT_DOCUMENT_PATH = Path("examples/sample_docs.md")
DEFAULT_EVAL_CASES_PATH = Path("examples/eval_cases.json")
UPLOAD_DIR = Path("output/uploads")
STREAM_DELTA_CHARS = 120
STREAM_DELTA_DELAY_SECONDS = 0.02
MONGO_STORE = MongoStore()


class QueryRequest(BaseModel):
    """API request for local document question answering."""

    query: str = Field(min_length=1)
    document_id: str | None = None
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
        allow_methods=["GET", "POST", "OPTIONS"],
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

    @app.get("/documents/{document_id}")
    def document_status(document_id: str) -> dict[str, Any]:
        try:
            document = DocumentRepository.from_store(MONGO_STORE).get(document_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if document is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        return document_status_payload(document).model_dump()

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
        result = run_query_request(request)
        metrics_registry.record_run(result.metrics)

        def events():
            yield _sse("planning", {"selected_agents": result.plan.selected_agents, "tasks": result.plan.tasks})
            yield _sse("retrieval", {"count": len(result.sources), "sources": [source_payload(source) for source in result.sources]})
            yield _sse("agents", {"agents": [agent_payload(agent) for agent in result.agents]})
            yield _sse("judge", result.grounding.__dict__)
            for delta in answer_deltas(result.answer):
                yield _sse("answer_delta", {"delta": delta})
                sleep(STREAM_DELTA_DELAY_SECONDS)
            yield _sse("final", workflow_payload(result))

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def run_query_request(request: QueryRequest) -> WorkflowResult:
    if request.document_id:
        return safe_run_document_query(
            request.query,
            request.document_id,
            request.orchestrator,
            request.require_llm_answer,
        )
    return safe_run_query(
        request.query,
        Path(request.document_path),
        request.orchestrator,
        request.retrieval_backend,
        request.require_llm_answer,
    )


def run_query(
    query: str,
    document_path: Path,
    orchestrator: str | None = None,
    retrieval_backend: str | None = None,
    require_llm_answer: bool = False,
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
        ).run(query)
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()


def run_document_query(
    query: str,
    document_id: str,
    orchestrator: str | None = None,
    require_llm_answer: bool = False,
) -> WorkflowResult:
    document = DocumentRepository.from_store(MONGO_STORE).get(document_id)
    if document is None:
        raise ValueError(f"Document not found: {document_id}")
    if document.status is not DocumentStatus.COMPLETED:
        raise RuntimeError(f"Document is not ready for queries: {document.status.value}")

    retriever = create_document_retriever(document_id)
    try:
        return create_workflow(
            retriever,
            orchestrator=orchestrator,
            require_llm_answer=require_llm_answer,
            default_answer_provider="ollama" if require_llm_answer else None,
        ).run(query)
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
    return DocumentStatusResponse(
        document_id=document.document_id,
        filename=document.title,
        document_path=document.file_path,
        status=document.status.value,
        progress_percentage=document.progress_percentage,
        current_stage=document.current_stage,
        stage_details=document.stage_details,
    )


def safe_run_query(
    query: str,
    document_path: Path,
    orchestrator: str | None = None,
    retrieval_backend: str | None = None,
    require_llm_answer: bool = False,
) -> WorkflowResult:
    try:
        return run_query(
            query,
            document_path,
            orchestrator=orchestrator,
            retrieval_backend=retrieval_backend,
            require_llm_answer=require_llm_answer,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def safe_run_document_query(
    query: str,
    document_id: str,
    orchestrator: str | None = None,
    require_llm_answer: bool = False,
) -> WorkflowResult:
    try:
        return run_document_query(
            query,
            document_id,
            orchestrator=orchestrator,
            require_llm_answer=require_llm_answer,
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
        "sources": [source_payload(source) for source in result.sources],
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


def source_payload(source: SearchResult) -> dict[str, Any]:
    return {
        "chunk_id": source.chunk.chunk_id,
        "document_id": source.chunk.document_id,
        "title": source.chunk.metadata.get("title"),
        "chunk_type": source.chunk.chunk_type.value,
        "score": source.score,
        "retrieval_type": source.retrieval_type.value,
        "highlights": source.highlights,
        "text": source.chunk.text,
    }


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def answer_deltas(answer: str, max_chars: int = STREAM_DELTA_CHARS) -> list[str]:
    chunks = []
    remaining = answer
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = max_chars
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    return chunks


app = build_app()
