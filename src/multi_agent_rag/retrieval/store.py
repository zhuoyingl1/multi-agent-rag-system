"""Local in-memory indexes for hybrid retrieval."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from multi_agent_rag.models import Chunk, RetrievalType, SearchResult
from multi_agent_rag.retrieval.tokenization import extract_entities, token_counts, tokenize


class LocalHybridStore:
    """In-memory keyword, vector-like, and entity indexes for local demos."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._keyword_index: dict[str, set[str]] = defaultdict(set)
        self._entity_index: dict[str, set[str]] = defaultdict(set)
        self._chunk_by_id: dict[str, Chunk] = {}
        self._vectors: dict[str, Counter[str]] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._remove_existing(chunk.chunk_id)
            self.chunks.append(chunk)
            self._chunk_by_id[chunk.chunk_id] = chunk
            counts = token_counts(chunk.text)
            self._vectors[chunk.chunk_id] = counts
            for token in counts:
                self._keyword_index[token].add(chunk.chunk_id)
            for entity in extract_entities(chunk.text):
                self._entity_index[entity].add(chunk.chunk_id)

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores: Counter[str] = Counter()
        for token in query_tokens:
            for chunk_id in self._keyword_index.get(token, set()):
                scores[chunk_id] += 1
        return self._results(scores, RetrievalType.KEYWORD, top_k, query_tokens)

    def vector_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = token_counts(query)
        if not query_vector:
            return []
        scores: dict[str, float] = {}
        query_norm = self._norm(query_vector)
        for chunk_id, vector in self._vectors.items():
            denominator = query_norm * self._norm(vector)
            if denominator == 0:
                continue
            score = sum(query_vector[token] * vector.get(token, 0) for token in query_vector) / denominator
            if score > 0:
                scores[chunk_id] = score
        return self._results(scores, RetrievalType.VECTOR, top_k, list(query_vector))

    def entity_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        entities = extract_entities(query)
        if not entities:
            return []
        scores: Counter[str] = Counter()
        for entity in entities:
            for chunk_id in self._entity_index.get(entity, set()):
                scores[chunk_id] += 1
        return self._results(scores, RetrievalType.ENTITY, top_k, sorted(entities))

    def _remove_existing(self, chunk_id: str) -> None:
        if chunk_id not in self._chunk_by_id:
            return
        self.chunks = [chunk for chunk in self.chunks if chunk.chunk_id != chunk_id]
        self._chunk_by_id.pop(chunk_id, None)
        self._vectors.pop(chunk_id, None)
        for index in (self._keyword_index, self._entity_index):
            for ids in index.values():
                ids.discard(chunk_id)

    def _results(
        self,
        scores: Counter[str] | dict[str, float],
        retrieval_type: RetrievalType,
        top_k: int,
        query_terms: list[str],
    ) -> list[SearchResult]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk=self._chunk_by_id[chunk_id],
                score=float(score),
                retrieval_type=retrieval_type,
                highlights=self._highlights(self._chunk_by_id[chunk_id].text, query_terms),
            )
            for chunk_id, score in ordered
            if chunk_id in self._chunk_by_id
        ]

    def _highlights(self, text: str, query_terms: list[str]) -> list[str]:
        lowered = text.lower()
        matched = []
        for term in query_terms:
            if term.lower() in lowered and term.lower() not in matched:
                matched.append(term.lower())
        return matched[:5]

    def _norm(self, counts: Counter[str]) -> float:
        return math.sqrt(sum(value * value for value in counts.values()))
