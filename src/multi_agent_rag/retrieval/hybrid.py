"""High-level local hybrid retriever."""

from __future__ import annotations

from multi_agent_rag.models import Chunk, SearchResult
from multi_agent_rag.retrieval.store import LocalHybridStore


class HybridRetriever:
    """Combine local keyword, vector-like, and entity retrieval signals."""

    def __init__(self, store: LocalHybridStore | None = None, top_k: int = 5) -> None:
        self.store = store or LocalHybridStore()
        self.top_k = top_k

    def index(self, chunks: list[Chunk]) -> None:
        self.store.add_chunks(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        limit = top_k or self.top_k
        candidates = [
            *self.store.keyword_search(query, top_k=limit),
            *self.store.vector_search(query, top_k=limit),
            *self.store.entity_search(query, top_k=limit),
        ]
        return self._dedupe(candidates)[:limit]

    def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
        by_chunk: dict[str, SearchResult] = {}
        for result in results:
            current = by_chunk.get(result.chunk.chunk_id)
            if current is None or result.score > current.score:
                by_chunk[result.chunk.chunk_id] = result
        return sorted(by_chunk.values(), key=lambda item: item.score, reverse=True)
