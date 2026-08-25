"""FastAPI application for the local multi-agent RAG prototype."""

from __future__ import annotations

import json
import re
from uuid import uuid4
from pathlib import Path
from time import sleep
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from multi_agent_rag.documents import load_document
from multi_agent_rag.documents import SUPPORTED_EXTENSIONS
from multi_agent_rag.evaluation import EvalReport, run_evaluation
from multi_agent_rag.integrations import check_integrations
from multi_agent_rag.models import AgentResult, SearchResult, WorkflowResult
from multi_agent_rag.observability import metrics_registry
from multi_agent_rag.orchestration import create_workflow
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever

DEFAULT_DOCUMENT_PATH = Path("examples/sample_docs.md")
DEFAULT_EVAL_CASES_PATH = Path("examples/eval_cases.json")
UPLOAD_DIR = Path("output/uploads")
STREAM_DELTA_CHARS = 120
STREAM_DELTA_DELAY_SECONDS = 0.02


class QueryRequest(BaseModel):
    """API request for local document question answering."""

    query: str = Field(min_length=1)
    document_path: str = Field(default=str(DEFAULT_DOCUMENT_PATH), min_length=1)
    orchestrator: str = Field(default="auto", pattern="^(auto|local|langgraph)$")


class EvaluationRequest(BaseModel):
    """API request for deterministic local evaluation."""

    document_path: str = Field(default=str(DEFAULT_DOCUMENT_PATH), min_length=1)
    cases_path: str = Field(default=str(DEFAULT_EVAL_CASES_PATH), min_length=1)
    orchestrator: str = Field(default="auto", pattern="^(auto|local|langgraph)$")


class UploadResponse(BaseModel):
    """API response for uploaded local documents."""

    filename: str
    document_path: str
    content_type: str | None
    size_bytes: int


def build_app() -> FastAPI:
    app = FastAPI(title="Multi-Agent RAG System V2", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
        return check_integrations().to_dict()

    @app.post("/documents/upload")
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        return (await save_uploaded_document(file)).model_dump()

    @app.post("/evaluate")
    def evaluate(request: EvaluationRequest) -> dict[str, Any]:
        return safe_run_evaluation(Path(request.document_path), Path(request.cases_path), request.orchestrator).to_dict()

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        result = safe_run_query(request.query, Path(request.document_path), request.orchestrator)
        metrics_registry.record_run(result.metrics)
        return workflow_payload(result)

    @app.post("/query/stream")
    def query_stream(request: QueryRequest) -> StreamingResponse:
        result = safe_run_query(request.query, Path(request.document_path), request.orchestrator)
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


def run_query(query: str, document_path: Path, orchestrator: str | None = None) -> WorkflowResult:
    document = load_document(document_path)
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    return create_workflow(retriever, orchestrator=orchestrator).run(query)


async def save_uploaded_document(file: UploadFile) -> UploadResponse:
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
    return UploadResponse(
        filename=filename,
        document_path=str(saved_path),
        content_type=file.content_type,
        size_bytes=len(content),
    )


def safe_run_query(query: str, document_path: Path, orchestrator: str | None = None) -> WorkflowResult:
    try:
        return run_query(query, document_path, orchestrator=orchestrator)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def safe_run_evaluation(document_path: Path, cases_path: Path, orchestrator: str | None = None) -> EvalReport:
    if not cases_path.exists():
        raise HTTPException(status_code=400, detail=f"Evaluation cases not found: {cases_path}")
    try:
        return run_evaluation(document_path, cases_path, orchestrator=orchestrator)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def workflow_payload(result: WorkflowResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "answer": result.answer,
        "plan": result.plan.__dict__,
        "agents": [agent_payload(agent) for agent in result.agents],
        "grounding": result.grounding.__dict__,
        "sources": [source_payload(source) for source in result.sources],
        "metrics": result.metrics,
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
