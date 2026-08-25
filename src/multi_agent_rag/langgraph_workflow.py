"""Optional LangGraph-backed orchestration for the RAG workflow."""

from __future__ import annotations

from time import perf_counter
from typing import Any, TypedDict

from multi_agent_rag.agents.coordinator import CoordinatorAgent
from multi_agent_rag.agents.experts import ExpertAgent
from multi_agent_rag.agents.judge import GroundingJudge
from multi_agent_rag.agents.planner import PlannerAgent
from multi_agent_rag.agents.summarizer import SummarizerAgent
from multi_agent_rag.models import AgentPlan, AgentResult, Coordination, JudgeResult, SearchResult, WorkflowResult
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import has_enough_evidence


class LangGraphState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    query: str
    started: float
    plan: AgentPlan
    sources: list[SearchResult]
    evidence_sufficient: bool
    coordination: Coordination
    agents: list[AgentResult]
    grounding: JudgeResult
    answer: str
    result: WorkflowResult


class LangGraphRAGWorkflow:
    """Run the same agent stages through a LangGraph state graph."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.planner = PlannerAgent()
        self.coordinator = CoordinatorAgent()
        self.judge = GroundingJudge()
        self.summarizer = SummarizerAgent()
        self.graph = self._build_graph()

    def run(self, query: str) -> WorkflowResult:
        final_state = self.graph.invoke({"query": query, "started": perf_counter()})
        return final_state["result"]

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("LangGraph workflow requires the 'langgraph' package. Install the production extra first.") from exc

        graph = StateGraph(LangGraphState)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("fallback", self._fallback)
        graph.add_node("coordinate", self._coordinate)
        graph.add_node("run_agents", self._run_agents)
        graph.add_node("judge", self._judge)
        graph.add_node("summarize", self._summarize)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {
                "sufficient": "coordinate",
                "insufficient": "fallback",
            },
        )
        graph.add_edge("fallback", END)
        graph.add_edge("coordinate", "run_agents")
        graph.add_edge("run_agents", "judge")
        graph.add_edge("judge", "summarize")
        graph.add_edge("summarize", END)
        return graph.compile()

    def _plan(self, state: LangGraphState) -> LangGraphState:
        return {"plan": self.planner.plan(state["query"])}

    def _retrieve(self, state: LangGraphState) -> LangGraphState:
        sources = self.retriever.retrieve(state["query"], top_k=self.top_k)
        return {"sources": sources, "evidence_sufficient": has_enough_evidence(state["query"], sources)}

    def _route_after_retrieval(self, state: LangGraphState) -> str:
        return "sufficient" if state["evidence_sufficient"] else "insufficient"

    def _fallback(self, state: LangGraphState) -> LangGraphState:
        sources = state.get("sources", [])
        grounding = self.judge.judge([], [])
        answer = self.summarizer.summarize(state["query"], [], grounding, [])
        result = self._result(
            state=state,
            agents=[],
            grounding=grounding,
            answer=answer,
            sources=[],
            candidate_sources=len(sources),
            evidence_status="insufficient",
        )
        return {"agents": [], "grounding": grounding, "answer": answer, "result": result}

    def _coordinate(self, state: LangGraphState) -> LangGraphState:
        return {"coordination": self.coordinator.coordinate(state["plan"], state["sources"])}

    def _run_agents(self, state: LangGraphState) -> LangGraphState:
        coordination = state["coordination"]
        agent_results = [
            ExpertAgent(agent_name).run(coordination.tasks[agent_name], coordination.sources)
            for agent_name in coordination.selected_agents
        ]
        return {"agents": agent_results}

    def _judge(self, state: LangGraphState) -> LangGraphState:
        return {"grounding": self.judge.judge(state["agents"], state["sources"])}

    def _summarize(self, state: LangGraphState) -> LangGraphState:
        answer = self.summarizer.summarize(state["query"], state["agents"], state["grounding"], state["sources"])
        result = self._result(
            state=state,
            agents=state["agents"],
            grounding=state["grounding"],
            answer=answer,
            sources=state["sources"],
            candidate_sources=len(state["sources"]),
            evidence_status="sufficient",
        )
        return {"answer": answer, "result": result}

    def _result(
        self,
        state: LangGraphState,
        agents: list[AgentResult],
        grounding: JudgeResult,
        answer: str,
        sources: list[SearchResult],
        candidate_sources: int,
        evidence_status: str,
    ) -> WorkflowResult:
        latency_ms = round((perf_counter() - state["started"]) * 1000, 2)
        metrics: dict[str, float | int | str] = {
            "selected_agents": len(state["plan"].selected_agents),
            "retrieved_sources": len(sources),
            "candidate_sources": candidate_sources,
            "completed_agents": len([result for result in agents if result.error is None]),
            "failed_agents": len([result for result in agents if result.error is not None]),
            "grounding_score": grounding.score,
            "latency_ms": latency_ms,
            "mode": "langgraph",
            "evidence_status": evidence_status,
        }
        return WorkflowResult(
            query=state["query"],
            answer=answer,
            plan=state["plan"],
            agents=agents,
            grounding=grounding,
            sources=sources,
            metrics=metrics,
        )
