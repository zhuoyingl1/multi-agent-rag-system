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


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "documents"
DEFAULT_QDRANT_DOCUMENT_COLLECTION = "document_chunks"
DEFAULT_RERANKER_CANDIDATE_MULTIPLIER = 3


def create_retriever(backend: str | None = None, top_k: int = 5) -> Retriever:
    selected = (backend or os.getenv("RETRIEVAL_BACKEND") or "qdrant").lower()
    if selected == "local":
        retriever: Retriever = HybridRetriever(top_k=top_k)
    elif selected == "qdrant":
        url = os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
        collection = os.getenv("QDRANT_COLLECTION") or DEFAULT_QDRANT_COLLECTION
        from multi_agent_rag.retrieval.qdrant_adapter import QdrantRetriever

        retriever = QdrantRetriever(url=url, collection=collection, top_k=top_k)
    else:
        raise ValueError("Retrieval backend must be one of: local, qdrant")

    return _with_optional_reranker(retriever)


def create_document_retriever(document_id: str, top_k: int = 5) -> Retriever:
    """Create a retriever bound to one persistently indexed document."""

    return create_document_scope_retriever([document_id], top_k)


def create_document_scope_retriever(document_ids: list[str], top_k: int = 5) -> Retriever:
    """Create a retriever bound to a persistent multi-document scope."""

    unique_ids = list(dict.fromkeys(document_ids))
    if not unique_ids:
        raise ValueError("A document retrieval scope must not be empty.")

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
    vector = QdrantDocumentRetriever(index, unique_ids, top_k=top_k)
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
        rewrite_enabled=(os.getenv("QUERY_REWRITE_ENABLED") or "true").lower() in {"1", "true", "yes", "on"},
        max_variants=int(os.getenv("QUERY_REWRITE_MAX_VARIANTS") or "3"),
    )
    retriever = PersistentHybridRetriever(
        vector,
        keyword,
        store,
        graph=graph,
        query_planner=query_planner,
        top_k=top_k,
        rrf_k=float(os.getenv("RRF_K") or "60"),
    )
    return _with_optional_reranker(retriever)


def _with_optional_reranker(retriever: Retriever) -> Retriever:
    reranker_model = os.getenv("RERANKER_MODEL")
    if not reranker_model:
        return retriever

    from multi_agent_rag.retrieval.reranking import RerankingRetriever, create_reranker

    candidate_multiplier = int(os.getenv("RERANKER_CANDIDATE_MULTIPLIER", str(DEFAULT_RERANKER_CANDIDATE_MULTIPLIER)))
    return RerankingRetriever(
        retriever=retriever,
        reranker=create_reranker(reranker_model),
        candidate_multiplier=candidate_multiplier,
        selection_policy=AdaptiveTopKPolicy(
            enabled=(os.getenv("DYNAMIC_TOP_K_ENABLED") or "true").lower() in {"1", "true", "yes", "on"},
            min_results=int(os.getenv("DYNAMIC_TOP_K_MIN") or "3"),
            max_results=int(os.getenv("DYNAMIC_TOP_K_MAX") or "8"),
            high_gap=float(os.getenv("DYNAMIC_TOP_K_GAP_HIGH") or "2.0"),
            low_gap=float(os.getenv("DYNAMIC_TOP_K_GAP_LOW") or "0.6"),
            context_budget_tokens=int(os.getenv("RAG_CONTEXT_BUDGET_TOKENS") or "1600"),
        ),
    )
