"""Retriever selection helpers for local and Qdrant-backed retrieval."""

from __future__ import annotations

import os
from typing import Protocol

from multi_agent_rag.models import Chunk, SearchResult
from multi_agent_rag.retrieval.hybrid import HybridRetriever


class Retriever(Protocol):
    """Shared retriever interface for local and production retrieval backends."""

    def index(self, chunks: list[Chunk]) -> None:
        """Index document chunks."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve relevant chunks."""


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "documents"


def create_retriever(backend: str | None = None, top_k: int = 5) -> Retriever:
    selected = (backend or os.getenv("RETRIEVAL_BACKEND") or "qdrant").lower()
    if selected == "local":
        return HybridRetriever(top_k=top_k)
    if selected == "qdrant":
        url = os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
        collection = os.getenv("QDRANT_COLLECTION") or DEFAULT_QDRANT_COLLECTION
        from multi_agent_rag.retrieval.qdrant_adapter import QdrantRetriever

        return QdrantRetriever(url=url, collection=collection, top_k=top_k)
    raise ValueError("Retrieval backend must be one of: local, qdrant")
