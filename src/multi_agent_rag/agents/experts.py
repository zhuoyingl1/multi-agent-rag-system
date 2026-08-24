"""Specialist expert agents for deterministic evidence analysis."""

from __future__ import annotations

import re

from multi_agent_rag.models import AgentResult, SearchResult
from multi_agent_rag.retrieval.tokenization import tokenize

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


class ExpertAgent:
    """Produce source-grounded findings for one specialist role."""

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

        query_terms = set(tokenize(task))
        evidence_points = self._select_evidence_points(sources, query_terms, limit=3)
        role_label = self.name.replace("_", " ").title()
        content = f"{role_label} agent finding: " + " ".join(evidence_points)
        confidence = min(0.95, 0.45 + 0.12 * len(sources) + 0.05 * len(evidence_points))
        return AgentResult(
            agent_name=self.name,
            task=task,
            content=content,
            confidence=round(confidence, 2),
            sources=sources[:3],
        )

    def _select_evidence_points(self, sources: list[SearchResult], query_terms: set[str], limit: int) -> list[str]:
        candidates: list[tuple[int, str]] = []
        for source in sources:
            for sentence in self._sentences(source.chunk.text):
                terms = set(tokenize(sentence))
                score = len(terms & query_terms)
                if score > 0:
                    candidates.append((score, sentence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        for _score, sentence in candidates:
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) >= limit:
                return selected
        if selected:
            return selected

        for source in sources:
            fallback = self._focused_snippet(source.chunk.text, query_terms)
            if fallback not in selected:
                selected.append(fallback)
            if len(selected) >= limit:
                break
        return selected

    def _sentences(self, text: str) -> list[str]:
        compact = " ".join(text.split())
        pieces = [piece.strip() for piece in SENTENCE_PATTERN.split(compact) if piece.strip()]
        return [self._snippet(piece, max_chars=220) for piece in pieces]

    def _focused_snippet(self, text: str, query_terms: set[str], max_chars: int = 220) -> str:
        compact = " ".join(text.split())
        lowered = compact.lower()
        positions = [lowered.find(term.lower()) for term in query_terms if term and lowered.find(term.lower()) >= 0]
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

    def _snippet(self, text: str, max_chars: int = 180) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

