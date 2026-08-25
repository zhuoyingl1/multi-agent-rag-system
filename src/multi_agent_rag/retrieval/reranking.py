"""Optional reranking layer for retrieved candidates."""

from __future__ import annotations

from typing import Protocol

from multi_agent_rag.models import Chunk, RetrievalType, SearchResult
from multi_agent_rag.retrieval.tokenization import tokenize


class Retriever(Protocol):
    """Minimal retriever interface used by the reranking wrapper."""

    def index(self, chunks: list[Chunk]) -> None:
        """Index document chunks."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve candidates."""


class Reranker(Protocol):
    """Shared reranker interface."""

    name: str

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Return reranked results."""


class LexicalReranker:
    """Small deterministic reranker used for local tests and demos."""

    name = "lexical"

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        terms = set(tokenize(query))
        scored = [self._reranked_result(result, terms) for result in results]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _reranked_result(self, result: SearchResult, terms: set[str]) -> SearchResult:
        text = result.chunk.text.lower()
        matched = [term for term in terms if term in text]
        coverage = len(matched) / max(1, len(terms))
        score = round((coverage * 2.0) + (result.score * 0.1), 4)
        highlights = list(dict.fromkeys([*result.highlights, *matched[:5], "reranked"]))
        return SearchResult(
            chunk=result.chunk,
            score=score,
            retrieval_type=RetrievalType.RERANKED,
            highlights=highlights[:8],
        )


class SentenceTransformerReranker:
    """Cross-encoder reranker backed by sentence-transformers models."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "BGE reranking requires sentence-transformers. Install the production extra first."
            ) from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        scores = self.model.predict(pairs)
        reranked = [
            SearchResult(
                chunk=result.chunk,
                score=round(float(score), 4),
                retrieval_type=RetrievalType.RERANKED,
                highlights=list(dict.fromkeys([*result.highlights, "reranked"])),
            )
            for result, score in zip(results, scores)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


class RerankingRetriever:
    """Retriever wrapper that reranks a larger candidate pool."""

    def __init__(self, retriever: Retriever, reranker: Reranker, candidate_multiplier: int = 3) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.last_candidate_count = 0
        self.last_reranker = reranker.name

    def index(self, chunks: list[Chunk]) -> None:
        self.retriever.index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        limit = top_k or 5
        candidate_limit = limit * self.candidate_multiplier
        candidates = self.retriever.retrieve(query, top_k=candidate_limit)
        self.last_candidate_count = len(candidates)
        return self.reranker.rerank(query, candidates, limit)

    def close(self) -> None:
        close = getattr(self.retriever, "close", None)
        if callable(close):
            close()


def create_reranker(model_name: str) -> Reranker:
    """Create a reranker from configuration."""

    if model_name.lower() in {"local", "lexical", "deterministic"}:
        return LexicalReranker()
    return SentenceTransformerReranker(model_name)
