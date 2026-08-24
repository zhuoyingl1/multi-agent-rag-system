"""Final answer composer for the local workflow."""

from __future__ import annotations

from multi_agent_rag.models import AgentResult, JudgeResult, SearchResult


class SummarizerAgent:
    """Compose a readable final answer from specialist findings."""

    def summarize(self, query: str, agent_results: list[AgentResult], judge: JudgeResult, sources: list[SearchResult]) -> str:
        successful = [result for result in agent_results if not result.error]
        if not successful:
            finding_lines = ["- The workflow could not produce a grounded finding from retrieved evidence."]
        else:
            finding_lines = [f"- {result.content}" for result in successful]

        evidence_lines = []
        for source in sources[:4]:
            title = source.chunk.metadata.get("title", source.chunk.document_id)
            snippet = self._focused_snippet(source)
            evidence_lines.append(f"- {title}: {snippet}")
        if not evidence_lines:
            evidence_lines.append("- No retrieved evidence was available.")

        source_titles = self._source_titles(sources)
        source_text = ", ".join(source_titles[:3]) if source_titles else "no retrieved sources"
        unsupported = "; ".join(judge.unsupported_claims) if judge.unsupported_claims else "None"

        return "\n".join(
            [
                f"Question: {query}",
                "",
                "Answer:",
                *finding_lines,
                "",
                "Evidence:",
                *evidence_lines,
                "",
                f"Grounding score: {judge.score}",
                f"Unsupported claims: {unsupported}",
                f"Sources: {source_text}",
            ]
        )

    def _source_titles(self, sources: list[SearchResult]) -> list[str]:
        titles = []
        for source in sources:
            title = source.chunk.metadata.get("title", source.chunk.document_id)
            if title not in titles:
                titles.append(title)
        return titles

    def _focused_snippet(self, source: SearchResult, max_chars: int = 220) -> str:
        compact = " ".join(source.chunk.text.split())
        lowered = compact.lower()
        positions = [lowered.find(term.lower()) for term in source.highlights if term and lowered.find(term.lower()) >= 0]
        if not positions:
            return self._snippet(compact, max_chars=max_chars)
        start = max(0, min(positions) - max_chars // 3)
        end = min(len(compact), start + max_chars)
        snippet = compact[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(compact):
            snippet = snippet.rstrip() + "..."
        return snippet

    def _snippet(self, text: str, max_chars: int = 220) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."
