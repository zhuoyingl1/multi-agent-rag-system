"""Coordinator agent for assigning evidence to specialists."""

from __future__ import annotations

from multi_agent_rag.models import AgentPlan, Coordination, SearchResult


class CoordinatorAgent:
    """Bind a planner decision to the retrieved evidence set."""

    def coordinate(self, plan: AgentPlan, sources: list[SearchResult]) -> Coordination:
        reasoning = "Assigned the same top evidence set to each selected specialist for a transparent local demo."
        return Coordination(
            selected_agents=plan.selected_agents,
            tasks=plan.tasks,
            sources=sources,
            evidence_count=len(sources),
            reasoning=reasoning,
        )
