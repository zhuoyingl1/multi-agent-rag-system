"""Shared retrieval protocols without concrete adapter imports."""

from typing import Protocol

from multi_agent_rag.models import Chunk, SearchResult


class Retriever(Protocol):
    """Shared interface for retrieval implementations."""

    def index(self, chunks: list[Chunk]) -> None:
        """Index document chunks."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve relevant chunks."""
