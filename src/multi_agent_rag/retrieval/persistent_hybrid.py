"""Persistent vector and keyword retrieval with reciprocal rank fusion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math

from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.persistence import ChunkRepository, MongoStore
from multi_agent_rag.retrieval.factory_types import Retriever
from multi_agent_rag.retrieval.query_planning import RetrievalQueryPlan, RetrievalQueryPlanner
from multi_agent_rag.retrieval.tokenization import tokenize


class MongoKeywordRetriever:
    """Rank persisted chunks from one or more documents with BM25."""

    def __init__(self, chunks: ChunkRepository, document_id: str | Sequence[str], top_k: int = 50) -> None:
        self.chunks = chunks
        self.document_ids = [document_id] if isinstance(document_id, str) else list(document_id)
        self.top_k = top_k

    def index(self, chunks: list[Chunk]) -> None:
        raise RuntimeError("Persistent document chunks must be indexed during ingestion.")

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        query_terms = tokenize(query)
        records = (
            self.chunks.list_for_document(self.document_ids[0])
            if len(self.document_ids) == 1
            else self.chunks.list_for_documents(self.document_ids)
        )
        if not query_terms or not records:
            return []

        tokenized = [(record, tokenize(record.text)) for record in records]
        average_length = sum(len(terms) for _, terms in tokenized) / len(tokenized)
        document_frequency: Counter[str] = Counter()
        for _, terms in tokenized:
            document_frequency.update(set(terms))

        results: list[SearchResult] = []
        for record, terms in tokenized:
            score = self._bm25(query_terms, terms, document_frequency, len(records), average_length)
            if score <= 0:
                continue
            chunk = Chunk(
                document_id=record.document_id,
                chunk_id=record.chunk_id,
                text=record.text,
                chunk_type=ChunkType(record.chunk_type),
                index=record.index,
                metadata={str(key): str(value) for key, value in record.metadata.items()},
            )
            highlights = [term for term in dict.fromkeys(query_terms) if term in set(terms)]
            results.append(SearchResult(chunk, score, RetrievalType.KEYWORD, highlights[:5]))
        return sorted(results, key=lambda item: item.score, reverse=True)[: top_k or self.top_k]

    def _bm25(
        self,
        query_terms: list[str],
        terms: list[str],
        document_frequency: Counter[str],
        document_count: int,
        average_length: float,
    ) -> float:
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            inverse_frequency = math.log(1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + 1.5 * (0.25 + 0.75 * len(terms) / max(average_length, 1))
            score += inverse_frequency * (frequency * 2.5) / denominator
        return score


@dataclass
class _FusedCandidate:
    result: SearchResult
    rank_score: float = 0.0
    modalities: set[RetrievalType] | None = None
    highlights: list[str] | None = None


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[SearchResult], float]],
    limit: int,
    rrf_k: float = 60.0,
) -> list[SearchResult]:
    """Fuse heterogeneous rankings while preserving raw relevance scores."""

    candidates: dict[str, _FusedCandidate] = {}
    for results, weight in ranked_lists:
        for rank, result in enumerate(results, start=1):
            candidate = candidates.setdefault(result.chunk.chunk_id, _FusedCandidate(result=result))
            candidate.rank_score += weight / (rrf_k + rank)
            candidate.modalities = candidate.modalities or set()
            candidate.modalities.add(result.retrieval_type)
            candidate.highlights = candidate.highlights or []
            candidate.highlights.extend(result.highlights)
            if result.score > candidate.result.score:
                candidate.result = result

    ordered = sorted(candidates.values(), key=lambda item: item.rank_score, reverse=True)[:limit]
    return [
        SearchResult(
            chunk=candidate.result.chunk,
            score=candidate.result.score,
            retrieval_type=(
                RetrievalType.HYBRID
                if candidate.modalities and len(candidate.modalities) > 1
                else candidate.result.retrieval_type
            ),
            highlights=list(dict.fromkeys(candidate.highlights or []))[:5],
        )
        for candidate in ordered
    ]


def merge_query_variant_results(groups: list[list[SearchResult]]) -> list[SearchResult]:
    """Keep the strongest result per chunk across rewritten query variants."""

    strongest: dict[str, SearchResult] = {}
    for group in groups:
        for result in group:
            current = strongest.get(result.chunk.chunk_id)
            if current is None or result.score > current.score:
                strongest[result.chunk.chunk_id] = result
    return sorted(strongest.values(), key=lambda item: item.score, reverse=True)


class PersistentHybridRetriever:
    """Run persistent vector, BM25, and graph retrieval concurrently."""

    def __init__(
        self,
        vector: Retriever,
        keyword: Retriever,
        store: MongoStore,
        graph: Retriever | None = None,
        query_planner: RetrievalQueryPlanner | None = None,
        top_k: int = 5,
        rrf_k: float = 60.0,
    ) -> None:
        self.vector = vector
        self.keyword = keyword
        self.graph = graph
        self.query_planner = query_planner or RetrievalQueryPlanner()
        self.store = store
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.last_candidate_count = 0
        self.last_errors: dict[str, str] = {}
        self.last_query_plan = RetrievalQueryPlan("general", False, [])

    def index(self, chunks: list[Chunk]) -> None:
        raise RuntimeError("Persistent document chunks must be indexed during ingestion.")

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        limit = top_k or self.top_k
        candidate_limit = max(50, limit * 3)
        self.last_query_plan = self.query_planner.plan(query)
        retrievers = [("vector", self.vector, 1.0), ("keyword", self.keyword, 0.8)]
        if self.graph is not None:
            retrievers.append(("graph", self.graph, 0.7))

        self.last_errors = {}
        grouped_results: dict[str, list[list[SearchResult]]] = {name: [] for name, _, _ in retrievers}
        tasks = [
            (name, variant_index, retriever, variant)
            for name, retriever, _ in retrievers
            for variant_index, variant in enumerate(self.last_query_plan.query_variants)
        ]
        with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as executor:
            futures = [
                (name, variant_index, executor.submit(retriever.retrieve, variant, candidate_limit))
                for name, variant_index, retriever, variant in tasks
            ]
            for name, variant_index, future in futures:
                try:
                    grouped_results[name].append(future.result())
                except Exception as exc:
                    self.last_errors[f"{name}[{variant_index}]"] = f"{exc.__class__.__name__}: {exc}"

        weights = {name: weight for name, _, weight in retrievers}
        ranked_lists = [
            (merge_query_variant_results(grouped_results[name]), weights[name])
            for name, _, _ in retrievers
        ]

        all_results = [result for results, _ in ranked_lists for result in results]
        self.last_candidate_count = len({result.chunk.chunk_id for result in all_results})
        return reciprocal_rank_fusion(
            ranked_lists,
            limit=limit,
            rrf_k=self.rrf_k,
        )

    def close(self) -> None:
        close = getattr(self.vector, "close", None)
        if callable(close):
            close()
        close = getattr(self.graph, "close", None)
        if callable(close):
            close()
        self.store.close()
