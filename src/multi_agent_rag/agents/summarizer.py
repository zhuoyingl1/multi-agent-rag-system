"""Final answer composer for the local workflow."""

from __future__ import annotations

from multi_agent_rag.models import AgentResult, JudgeResult, SearchResult


class SummarizerAgent:
    """Compose a readable final answer from specialist findings."""

    def summarize(self, query: str, agent_results: list[AgentResult], judge: JudgeResult, sources: list[SearchResult]) -> str:
        successful = [result for result in agent_results if not result.error]
        direct_answer = self._direct_answer(query, sources)
        if not successful:
            analysis_lines = ["- The workflow could not produce a grounded finding from retrieved evidence."]
        else:
            analysis_lines = [f"- {result.agent_name}: {result.content}" for result in successful]

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
                direct_answer,
                "",
                "Analysis:",
                *analysis_lines,
                "",
                "Evidence:",
                *evidence_lines,
                "",
                f"Grounding score: {judge.score}",
                f"Unsupported claims: {unsupported}",
                f"Sources: {source_text}",
            ]
        )

    def _direct_answer(self, query: str, sources: list[SearchResult]) -> str:
        if not sources:
            return "No sufficiently relevant retrieved evidence was available, so the workflow cannot provide a grounded direct answer."

        title = self._best_title(sources)
        matched_terms = self._matched_terms(query, sources)
        if matched_terms:
            term_text = ", ".join(matched_terms[:8])
            return f"The strongest retrieved match is {title}, supported by evidence mentioning {term_text}."
        return f"The strongest retrieved match is {title}, based on the highest-scoring retrieved evidence."

    def _best_title(self, sources: list[SearchResult]) -> str:
        best = sources[0]
        chunk_text = " ".join(best.chunk.text.split())
        title = best.chunk.metadata.get("title", best.chunk.document_id)
        project_name = self._project_name(chunk_text)
        if project_name:
            return project_name
        return title

    def _project_name(self, text: str) -> str | None:
        marker = "Projects "
        if marker in text:
            after_marker = text.split(marker, 1)[1].strip()
            date_index = self._first_date_index(after_marker)
            if date_index > 0:
                return after_marker[:date_index].strip()

        date_index = self._first_date_index(text)
        if date_index <= 0:
            return None
        before_date = text[:date_index].strip()
        words = before_date.split()
        if not words:
            return None
        return " ".join(words[-6:])

    def _first_date_index(self, text: str) -> int:
        for index in range(max(0, len(text) - 6)):
            window = text[index : index + 7]
            if self._looks_like_month_year_range(window):
                return index
        return -1

    def _looks_like_month_year_range(self, text: str) -> bool:
        if len(text) != 7:
            return False
        return (
            text[0].isdigit()
            and text[1].isdigit()
            and text[2] == "/"
            and text[3].isdigit()
            and text[4].isdigit()
            and text[5].isdigit()
            and text[6].isdigit()
        )

    def _matched_terms(self, query: str, sources: list[SearchResult]) -> list[str]:
        stopwords = {
            "and",
            "or",
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "which",
            "what",
            "how",
            "resume",
            "project",
            "projects",
            "mention",
            "mentions",
        }
        display_names = {
            "rag": "RAG",
            "fastapi": "FastAPI",
            "next.js": "Next.js",
            "next": "Next.js",
            "js": "JavaScript",
            "langgraph": "LangGraph",
            "qdrant": "Qdrant",
            "neo4j": "Neo4j",
        }
        query_terms = [term.strip(" ?.,:;!()[]{}\"'").lower() for term in query.replace("/", " ").split()]
        query_terms = [term for term in query_terms if len(term) > 1 and term not in stopwords]
        source_text = " ".join(source.chunk.text.lower() for source in sources)
        matched = []
        for term in query_terms:
            display = display_names.get(term, term)
            if term in source_text and display not in matched:
                matched.append(display)
        return matched

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
