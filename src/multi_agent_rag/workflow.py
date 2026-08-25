"""End-to-end deterministic multi-agent RAG workflow."""

from __future__ import annotations

from time import perf_counter

from multi_agent_rag.agents.coordinator import CoordinatorAgent
from multi_agent_rag.agents.experts import ExpertAgent
from multi_agent_rag.agents.judge import GroundingJudge
from multi_agent_rag.agents.planner import PlannerAgent
from multi_agent_rag.agents.summarizer import SummarizerAgent
from multi_agent_rag.models import AgentResult, SearchResult, WorkflowResult
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.retrieval.tokenization import tokenize

EVIDENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class MultiAgentRAGWorkflow:
    """Coordinate planning, retrieval, specialist analysis, judging, and summarization."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.planner = PlannerAgent()
        self.coordinator = CoordinatorAgent()
        self.judge = GroundingJudge()
        self.summarizer = SummarizerAgent()

    def run(self, query: str) -> WorkflowResult:
        started = perf_counter()
        plan = self.planner.plan(query)
        sources = self.retriever.retrieve(query, top_k=self.top_k)
        candidate_count = retriever_candidate_count(self.retriever, sources)
        reranker = retriever_reranker_name(self.retriever)
        if not self._has_enough_evidence(query, sources):
            grounding = self.judge.judge([], [])
            answer = self.summarizer.summarize(query, [], grounding, [])
            latency_ms = round((perf_counter() - started) * 1000, 2)
            metrics: dict[str, float | int | str] = {
                "selected_agents": len(plan.selected_agents),
                "retrieved_sources": 0,
                "candidate_sources": candidate_count,
                "completed_agents": 0,
                "failed_agents": 0,
                "grounding_score": grounding.score,
                "latency_ms": latency_ms,
                "mode": "deterministic_local",
                "evidence_status": "insufficient",
                "reranker": reranker,
            }
            return WorkflowResult(
                query=query,
                answer=answer,
                plan=plan,
                agents=[],
                grounding=grounding,
                sources=[],
                metrics=metrics,
            )

        coordination = self.coordinator.coordinate(plan, sources)

        agent_results: list[AgentResult] = []
        for agent_name in coordination.selected_agents:
            agent = ExpertAgent(agent_name)
            agent_results.append(agent.run(coordination.tasks[agent_name], coordination.sources))

        grounding = self.judge.judge(agent_results, sources)
        answer = self.summarizer.summarize(query, agent_results, grounding, sources)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        metrics: dict[str, float | int | str] = {
            "selected_agents": len(plan.selected_agents),
            "retrieved_sources": len(sources),
            "candidate_sources": candidate_count,
            "completed_agents": len([result for result in agent_results if result.error is None]),
            "failed_agents": len([result for result in agent_results if result.error is not None]),
            "grounding_score": grounding.score,
            "latency_ms": latency_ms,
            "mode": "deterministic_local",
            "evidence_status": "sufficient",
            "reranker": reranker,
        }
        return WorkflowResult(
            query=query,
            answer=answer,
            plan=plan,
            agents=agent_results,
            grounding=grounding,
            sources=sources,
            metrics=metrics,
        )

    def _has_enough_evidence(self, query: str, sources: list[SearchResult]) -> bool:
        return has_enough_evidence(query, sources)


def has_enough_evidence(query: str, sources: list[SearchResult]) -> bool:
    if not sources:
        return False
    query_terms = [term for term in tokenize(query) if term not in EVIDENCE_STOPWORDS]
    if not query_terms:
        return False

    source_text = " ".join(source.chunk.text for source in sources).lower()
    matched_terms = {term for term in query_terms if term in source_text}
    min_required = 1 if len(set(query_terms)) <= 2 else 2
    coverage = len(matched_terms) / max(1, len(set(query_terms)))
    return len(matched_terms) >= min_required and coverage >= 0.2


def retriever_candidate_count(retriever: object, sources: list[SearchResult]) -> int:
    return int(getattr(retriever, "last_candidate_count", len(sources)))


def retriever_reranker_name(retriever: object) -> str:
    return str(getattr(retriever, "last_reranker", "none"))
