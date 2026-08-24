"""Final answer composer for the local workflow."""

from __future__ import annotations

from multi_agent_rag.models import AgentResult, JudgeResult, SearchResult


class SummarizerAgent:
    """Compose a concise final answer from specialist findings."""

    def summarize(self, query: str, agent_results: list[AgentResult], judge: JudgeResult, sources: list[SearchResult]) -> str:
        if not agent_results:
            return "No specialist findings were produced."

        findings = " ".join(result.content for result in agent_results if not result.error)
        if not findings:
            findings = "The workflow could not produce a grounded finding from retrieved evidence."

        source_titles = []
        for source in sources:
            title = source.chunk.metadata.get("title", source.chunk.document_id)
            if title not in source_titles:
                source_titles.append(title)

        source_text = ", ".join(source_titles[:3]) if source_titles else "no retrieved sources"
        return (
            f"Question: {query}\n"
            f"Answer: {findings}\n"
            f"Grounding score: {judge.score}.\n"
            f"Sources: {source_text}."
        )
