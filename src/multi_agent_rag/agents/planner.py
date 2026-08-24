"""Planner agent for selecting specialist roles."""

from __future__ import annotations

from multi_agent_rag.models import AgentPlan
from multi_agent_rag.retrieval.tokenization import tokenize


class PlannerAgent:
    """Select specialist agents from simple query signals."""

    def plan(self, query: str) -> AgentPlan:
        tokens = set(tokenize(query))
        selected: list[str] = []

        if tokens & {"retrieve", "retrieval", "vector", "keyword", "graph", "neo4j", "qdrant", "evidence"}:
            selected.append("retrieval")
        if tokens & {"metric", "metrics", "evaluate", "evaluation", "grounding", "hallucination", "judge"}:
            selected.append("evaluation")
        if tokens & {"api", "fastapi", "sse", "frontend", "next", "code", "implementation", "timeout"}:
            selected.append("implementation")
        if not selected:
            selected.append("research")

        tasks = {name: self._task_for(name, query) for name in selected}
        reasoning = "Selected agents from query terms and kept the plan minimal."
        return AgentPlan(selected_agents=selected, tasks=tasks, reasoning=reasoning)

    def _task_for(self, name: str, query: str) -> str:
        task_map = {
            "retrieval": "Analyze the retrieved evidence and explain retrieval behavior.",
            "evaluation": "Assess grounding, source coverage, and hallucination risk.",
            "implementation": "Identify implementation details, APIs, and operational risks.",
            "research": "Synthesize the retrieved evidence into a concise research answer.",
        }
        return f"{task_map[name]} Query: {query}"
