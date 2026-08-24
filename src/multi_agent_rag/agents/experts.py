"""Specialist expert agents for deterministic evidence analysis."""

from __future__ import annotations

from multi_agent_rag.models import AgentResult, SearchResult


class ExpertAgent:
    """Produce a short source-grounded finding for one specialist role."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, task: str, sources: list[SearchResult]) -> AgentResult:
        if not sources:
            return AgentResult(
                agent_name=self.name,
                task=task,
                content="No retrieved evidence was available for this specialist.",
                confidence=0.1,
                sources=[],
                error="missing_evidence",
            )

        snippets = [self._snippet(source.chunk.text) for source in sources[:2]]
        content = f"{self.name.title()} agent finding: " + " ".join(snippets)
        confidence = min(0.95, 0.45 + 0.12 * len(sources))
        return AgentResult(
            agent_name=self.name,
            task=task,
            content=content,
            confidence=round(confidence, 2),
            sources=sources[:3],
        )

    def _snippet(self, text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= 180:
            return compact
        return compact[:177].rstrip() + "..."
