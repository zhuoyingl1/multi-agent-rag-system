"""Rank-based evaluation for document retrieval pipelines."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from multi_agent_rag.documents import load_document
from multi_agent_rag.models import Chunk
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.factory import create_retriever
from multi_agent_rag.retrieval.factory_types import Retriever


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A query with the chunk indices expected to contain its evidence."""

    case_id: str
    query: str
    relevant_chunk_indices: list[int]
    min_recall_at_k: float = 1.0


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Rank metrics and retrieved chunk details for one query."""

    case_id: str
    query: str
    passed: bool
    relevant_chunk_indices: list[int]
    retrieved_document_ids: list[str]
    retrieved_chunk_indices: list[int]
    relevant_ranks: list[int]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: float
    retrieval_types: list[str]


@dataclass(frozen=True)
class RetrievalEvalReport:
    """Aggregated rank metrics for a retrieval benchmark."""

    document_path: str
    retrieval_backend: str
    top_k: int
    document_chunk_count: int
    case_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    average_recall_at_k: float
    average_precision_at_k: float
    mean_reciprocal_rank: float
    average_ndcg_at_k: float
    average_latency_ms: float
    cases: list[RetrievalCaseResult]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def load_retrieval_eval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    """Load and validate rank-based retrieval cases from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Retrieval evaluation data must contain a cases list.")

    cases: list[RetrievalEvalCase] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each retrieval evaluation case must be an object.")
        case_id = str(item.get("case_id") or "").strip()
        query = str(item.get("query") or "").strip()
        raw_indices = item.get("relevant_chunk_indices")
        if not case_id or not query:
            raise ValueError("Each retrieval evaluation case requires a case_id and query.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate retrieval evaluation case_id: {case_id}")
        if not isinstance(raw_indices, list) or not raw_indices or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in raw_indices
        ):
            raise ValueError(f"Case {case_id} requires non-negative relevant_chunk_indices.")
        min_recall = float(item.get("min_recall_at_k", 1.0))
        if not 0.0 <= min_recall <= 1.0:
            raise ValueError(f"Case {case_id} min_recall_at_k must be between 0 and 1.")
        seen_ids.add(case_id)
        cases.append(
            RetrievalEvalCase(
                case_id=case_id,
                query=query,
                relevant_chunk_indices=sorted(set(raw_indices)),
                min_recall_at_k=min_recall,
            )
        )
    return cases


def run_retrieval_evaluation(
    document_path: str | Path,
    cases_path: str | Path,
    *,
    retrieval_backend: str | None = None,
    top_k: int = 5,
) -> RetrievalEvalReport:
    """Index one document and evaluate the retriever against ranked gold chunks."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    document = load_document(document_path)
    chunks = chunk_document(document)
    cases = load_retrieval_eval_cases(cases_path)
    _validate_gold_indices(cases, chunks)
    retriever = create_retriever(retrieval_backend, top_k=top_k)
    try:
        retriever.index(chunks)
        results = [
            _run_case(retriever, case, top_k, document.stable_id())
            for case in cases
        ]
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()

    passed_count = sum(result.passed for result in results)
    case_count = len(results)
    backend = (retrieval_backend or os.getenv("RETRIEVAL_BACKEND") or "qdrant").lower()
    return RetrievalEvalReport(
        document_path=str(document_path),
        retrieval_backend=backend,
        top_k=top_k,
        document_chunk_count=len(chunks),
        case_count=case_count,
        passed_count=passed_count,
        failed_count=case_count - passed_count,
        pass_rate=_ratio(passed_count, case_count),
        average_recall_at_k=_average([result.recall_at_k for result in results]),
        average_precision_at_k=_average([result.precision_at_k for result in results]),
        mean_reciprocal_rank=_average([result.reciprocal_rank for result in results]),
        average_ndcg_at_k=_average([result.ndcg_at_k for result in results]),
        average_latency_ms=_average([result.latency_ms for result in results]),
        cases=results,
    )


def _validate_gold_indices(cases: list[RetrievalEvalCase], chunks: list[Chunk]) -> None:
    available = {chunk.index for chunk in chunks}
    for case in cases:
        missing = sorted(set(case.relevant_chunk_indices) - available)
        if missing:
            values = ", ".join(str(index) for index in missing)
            raise ValueError(f"Case {case.case_id} references unavailable chunk indices: {values}")


def _run_case(
    retriever: Retriever,
    case: RetrievalEvalCase,
    top_k: int,
    document_id: str,
) -> RetrievalCaseResult:
    started_at = perf_counter()
    retrieved = retriever.retrieve(case.query, top_k=top_k)
    latency_ms = round((perf_counter() - started_at) * 1000, 4)
    relevant = set(case.relevant_chunk_indices)
    retrieved_document_ids = [result.chunk.document_id for result in retrieved]
    retrieved_indices = [result.chunk.index for result in retrieved]
    hit_flags = [
        result.chunk.document_id == document_id and result.chunk.index in relevant
        for result in retrieved
    ]
    relevant_ranks = [rank for rank, is_relevant in enumerate(hit_flags, start=1) if is_relevant]
    hit_count = len(relevant_ranks)
    recall = _ratio(hit_count, len(relevant))
    precision = _ratio(hit_count, top_k)
    reciprocal_rank = round(1.0 / relevant_ranks[0], 4) if relevant_ranks else 0.0
    ndcg = _ndcg_at_k(hit_flags, len(relevant), top_k)
    return RetrievalCaseResult(
        case_id=case.case_id,
        query=case.query,
        passed=recall >= case.min_recall_at_k,
        relevant_chunk_indices=case.relevant_chunk_indices,
        retrieved_document_ids=retrieved_document_ids,
        retrieved_chunk_indices=retrieved_indices,
        relevant_ranks=relevant_ranks,
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
        latency_ms=latency_ms,
        retrieval_types=[result.retrieval_type.value for result in retrieved],
    )


def _ndcg_at_k(hit_flags: list[bool], relevant_count: int, top_k: int) -> float:
    gains = [1.0 if is_relevant else 0.0 for is_relevant in hit_flags[:top_k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_hits = min(relevant_count, top_k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / ideal_dcg, 4) if ideal_dcg else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _average(values: list[float]) -> float:
    return round(float(mean(values)), 4) if values else 0.0
