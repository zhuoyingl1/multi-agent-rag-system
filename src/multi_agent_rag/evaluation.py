"""Deterministic evaluation runner for local RAG workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from multi_agent_rag.documents import load_document
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow


@dataclass(frozen=True)
class EvalCase:
    """A local evaluation case with answer and source expectations."""

    case_id: str
    query: str
    expected_terms: list[str]
    required_source_terms: list[str]
    min_grounding_score: float = 0.5


@dataclass(frozen=True)
class EvalCaseResult:
    """Evaluation result for one query case."""

    case_id: str
    query: str
    passed: bool
    grounding_score: float
    retrieved_sources: int
    latency_ms: float
    failed_agents: int
    expected_terms_found: list[str]
    missing_expected_terms: list[str]
    source_terms_found: list[str]
    missing_source_terms: list[str]
    answer: str


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation report."""

    document_path: str
    case_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    average_grounding_score: float
    average_latency_ms: float
    average_retrieved_sources: float
    total_failed_agents: int
    cases: list[EvalCaseResult]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    case_path = Path(path)
    data = json.loads(case_path.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    return [
        EvalCase(
            case_id=str(item["case_id"]),
            query=str(item["query"]),
            expected_terms=[str(term) for term in item.get("expected_terms", [])],
            required_source_terms=[str(term) for term in item.get("required_source_terms", [])],
            min_grounding_score=float(item.get("min_grounding_score", 0.5)),
        )
        for item in cases
    ]


def run_evaluation(document_path: str | Path, cases_path: str | Path) -> EvalReport:
    document = load_document(document_path)
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    workflow = MultiAgentRAGWorkflow(retriever)

    case_results = [_run_case(workflow, case) for case in load_eval_cases(cases_path)]
    passed_count = len([result for result in case_results if result.passed])
    case_count = len(case_results)
    return EvalReport(
        document_path=str(document_path),
        case_count=case_count,
        passed_count=passed_count,
        failed_count=case_count - passed_count,
        pass_rate=round(passed_count / case_count, 4) if case_count else 0.0,
        average_grounding_score=_average([result.grounding_score for result in case_results]),
        average_latency_ms=_average([result.latency_ms for result in case_results]),
        average_retrieved_sources=_average([result.retrieved_sources for result in case_results]),
        total_failed_agents=sum(result.failed_agents for result in case_results),
        cases=case_results,
    )


def _run_case(workflow: MultiAgentRAGWorkflow, case: EvalCase) -> EvalCaseResult:
    result = workflow.run(case.query)
    answer_text = result.answer.lower()
    source_text = " ".join(source.chunk.text for source in result.sources).lower()
    expected_found, expected_missing = _term_matches(answer_text, case.expected_terms)
    source_found, source_missing = _term_matches(source_text, case.required_source_terms)
    grounding_score = float(result.grounding.score)
    retrieved_sources = int(result.metrics.get("retrieved_sources", len(result.sources)))
    failed_agents = int(result.metrics.get("failed_agents", 0))
    latency_ms = float(result.metrics.get("latency_ms", 0.0))
    passed = (
        not expected_missing
        and not source_missing
        and grounding_score >= case.min_grounding_score
        and retrieved_sources > 0
        and failed_agents == 0
    )
    return EvalCaseResult(
        case_id=case.case_id,
        query=case.query,
        passed=passed,
        grounding_score=grounding_score,
        retrieved_sources=retrieved_sources,
        latency_ms=latency_ms,
        failed_agents=failed_agents,
        expected_terms_found=expected_found,
        missing_expected_terms=expected_missing,
        source_terms_found=source_found,
        missing_source_terms=source_missing,
        answer=result.answer,
    )


def _term_matches(text: str, terms: list[str]) -> tuple[list[str], list[str]]:
    found = []
    missing = []
    for term in terms:
        if term.lower() in text:
            found.append(term)
        else:
            missing.append(term)
    return found, missing


def _average(values: list[float | int]) -> float:
    return round(float(mean(values)), 4) if values else 0.0
