"""End-to-end deterministic multi-agent RAG workflow."""

from __future__ import annotations

from time import perf_counter

from multi_agent_rag.agents.coordinator import CoordinatorAgent
from multi_agent_rag.agents.experts import ExpertAgent
from multi_agent_rag.agents.judge import GroundingJudge
from multi_agent_rag.agents.planner import PlannerAgent
from multi_agent_rag.agents.summarizer import SummarizerAgent
from multi_agent_rag.models import AgentResult, WorkflowResult
from multi_agent_rag.retrieval.hybrid import HybridRetriever


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
            "completed_agents": len([result for result in agent_results if result.error is None]),
            "failed_agents": len([result for result in agent_results if result.error is not None]),
            "grounding_score": grounding.score,
            "latency_ms": latency_ms,
            "mode": "deterministic_local",
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
