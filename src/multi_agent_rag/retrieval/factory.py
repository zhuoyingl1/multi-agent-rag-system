"""Retriever selection helpers for local and Qdrant-backed retrieval."""

from __future__ import annotations

import os

from multi_agent_rag.persistence import ChunkRepository, MongoStore
from multi_agent_rag.retrieval.embeddings import OllamaEmbeddingService
from multi_agent_rag.retrieval.factory_types import Retriever
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.retrieval.persistent_hybrid import MongoKeywordRetriever, PersistentHybridRetriever
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
    vector = QdrantDocumentRetriever(index, document_id, top_k=top_k)
    store = MongoStore()
    keyword = MongoKeywordRetriever(ChunkRepository.from_store(store), document_id)
    retriever = PersistentHybridRetriever(
        vector,
        keyword,
        store,
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
    )
