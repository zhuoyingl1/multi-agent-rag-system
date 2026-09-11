"""Retriever selection helpers for local and Qdrant-backed retrieval."""

from __future__ import annotations

import os

from multi_agent_rag.persistence import ChunkRepository, MongoStore
from multi_agent_rag.retrieval.adaptive import AdaptiveTopKPolicy
from multi_agent_rag.retrieval.embeddings import OllamaEmbeddingService
from multi_agent_rag.retrieval.factory_types import Retriever
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.retrieval.neo4j_adapter import Neo4jGraphAdapter, Neo4jGraphRetriever
from multi_agent_rag.retrieval.persistent_hybrid import MongoKeywordRetriever, PersistentHybridRetriever
from multi_agent_rag.retrieval.query_planning import RetrievalQueryPlanner
from multi_agent_rag.retrieval.vector_index import QdrantDocumentIndex, QdrantDocumentRetriever
from multi_agent_rag.runtime_config import RuntimeSettings


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "documents"
DEFAULT_QDRANT_DOCUMENT_COLLECTION = "document_chunks"
def create_retriever(
    backend: str | None = None,
    top_k: int | None = None,
    runtime_settings: RuntimeSettings | None = None,
) -> Retriever:
    settings = runtime_settings or RuntimeSettings.from_env()
    result_limit = top_k or settings.top_k
    selected = (backend or os.getenv("RETRIEVAL_BACKEND") or "qdrant").lower()
    if selected == "local":
        retriever: Retriever = HybridRetriever(top_k=result_limit)
    elif selected == "qdrant":
        url = os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
        collection = os.getenv("QDRANT_COLLECTION") or DEFAULT_QDRANT_COLLECTION
        from multi_agent_rag.retrieval.qdrant_adapter import QdrantRetriever

        retriever = QdrantRetriever(url=url, collection=collection, top_k=result_limit)
    else:
        raise ValueError("Retrieval backend must be one of: local, qdrant")

    return _with_optional_reranker(retriever, settings)


def create_document_retriever(
    document_id: str,
    top_k: int | None = None,
    runtime_settings: RuntimeSettings | None = None,
) -> Retriever:
    """Create a retriever bound to one persistently indexed document."""

    return create_document_scope_retriever([document_id], top_k, runtime_settings)


def create_document_scope_retriever(
    document_ids: list[str],
    top_k: int | None = None,
    runtime_settings: RuntimeSettings | None = None,
) -> Retriever:
    """Create a retriever bound to a persistent multi-document scope."""

    unique_ids = list(dict.fromkeys(document_ids))
    if not unique_ids:
        raise ValueError("A document retrieval scope must not be empty.")
    settings = runtime_settings or RuntimeSettings.from_env()
    result_limit = top_k or settings.top_k

    embedder = OllamaEmbeddingService(
        base_url=os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434",
        model_name=os.getenv("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text",
        timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS") or "60"),
    )
    index = QdrantDocumentIndex(
        url=os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL,
        collection=os.getenv("QDRANT_DOCUMENT_COLLECTION") or DEFAULT_QDRANT_DOCUMENT_COLLECTION,
        embedder=embedder,
        score_threshold=float(os.getenv("QDRANT_SCORE_THRESHOLD") or "0.5"),
    )
    vector = QdrantDocumentRetriever(index, unique_ids, top_k=result_limit)
    store = MongoStore()
    chunks = ChunkRepository.from_store(store)
    keyword = MongoKeywordRetriever(chunks, unique_ids)
    graph_adapter = Neo4jGraphAdapter(
        uri=os.getenv("NEO4J_URI") or "bolt://localhost:7687",
        user=os.getenv("NEO4J_USER") or "neo4j",
        password=os.getenv("NEO4J_PASSWORD") or "password123",
        database=os.getenv("NEO4J_DATABASE") or "neo4j",
    )
    graph = Neo4jGraphRetriever(graph_adapter, chunks, unique_ids)
    query_planner = RetrievalQueryPlanner(
        rewrite_enabled=settings.query_rewrite_enabled,
        max_variants=settings.query_rewrite_max_variants,
    )
    retriever = PersistentHybridRetriever(
        vector,
        keyword,
        store,
        graph=graph,
        query_planner=query_planner,
        top_k=result_limit,
        rrf_k=settings.rrf_k,
    )
    return _with_optional_reranker(retriever, settings)


def _with_optional_reranker(retriever: Retriever, settings: RuntimeSettings) -> Retriever:
    reranker_model = os.getenv("RERANKER_MODEL")
    if not reranker_model:
        return retriever

    from multi_agent_rag.retrieval.reranking import RerankingRetriever, create_reranker

    return RerankingRetriever(
        retriever=retriever,
        reranker=create_reranker(reranker_model),
        candidate_multiplier=settings.reranker_candidate_multiplier,
        selection_policy=AdaptiveTopKPolicy(
            enabled=settings.dynamic_top_k_enabled,
            min_results=settings.dynamic_top_k_min,
            max_results=settings.dynamic_top_k_max,
            high_gap=settings.dynamic_top_k_gap_high,
            low_gap=settings.dynamic_top_k_gap_low,
            context_budget_tokens=settings.context_budget_tokens,
        ),
    )
