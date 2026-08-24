"""FastAPI application for the local multi-agent RAG prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from multi_agent_rag.documents import load_document
from multi_agent_rag.models import SearchResult, WorkflowResult
from multi_agent_rag.observability import metrics_registry
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow

DEFAULT_DOCUMENT_PATH = Path("examples/sample_docs.md")


class QueryRequest(BaseModel):
    """API request for local document question answering."""

    query: str
    document_path: str = str(DEFAULT_DOCUMENT_PATH)


def build_app() -> FastAPI:
    app = FastAPI(title="Multi-Agent RAG System V2", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "mode": "deterministic_local",
            "default_document": str(DEFAULT_DOCUMENT_PATH),
        }

    @app.get("/health/metrics")
    def health_metrics() -> dict[str, Any]:
        return metrics_registry.snapshot()

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        result = run_query(request.query, Path(request.document_path))
        metrics_registry.record_run(result.metrics)
        return workflow_payload(result)

    @app.post("/query/stream")
    def query_stream(request: QueryRequest) -> StreamingResponse:
        def events():
            result = run_query(request.query, Path(request.document_path))
            metrics_registry.record_run(result.metrics)
            yield _sse("planning", {"selected_agents": result.plan.selected_agents, "tasks": result.plan.tasks})
            yield _sse("retrieval", {"count": len(result.sources), "sources": [source_payload(source) for source in result.sources]})
            yield _sse("agents", {"agents": [agent.__dict__ for agent in result.agents]})
            yield _sse("judge", result.grounding.__dict__)
            yield _sse("final", workflow_payload(result))

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def run_query(query: str, document_path: Path) -> WorkflowResult:
    document = load_document(document_path)
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    return MultiAgentRAGWorkflow(retriever).run(query)


def workflow_payload(result: WorkflowResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "answer": result.answer,
        "plan": result.plan.__dict__,
        "agents": [agent.__dict__ for agent in result.agents],
        "grounding": result.grounding.__dict__,
        "sources": [source_payload(source) for source in result.sources],
        "metrics": result.metrics,
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


app = build_app()
