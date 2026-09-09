"""Deterministic intent analysis and query rewriting for document retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from multi_agent_rag.retrieval.tokenization import tokenize


@dataclass(frozen=True)
class RetrievalQueryPlan:
    """A compact retrieval plan derived from one user question."""

    intent: str
    need_rewrite: bool
    query_variants: list[str]


class RetrievalQueryPlanner:
    """Classify retrieval intent and build a small set of query variants."""

    def __init__(self, rewrite_enabled: bool = True, max_variants: int = 3) -> None:
        self.rewrite_enabled = rewrite_enabled
        self.max_variants = max(1, max_variants)

    def plan(self, query: str) -> RetrievalQueryPlan:
        normalized = " ".join(query.split())
        intent = self._intent(normalized)
        need_rewrite = self.rewrite_enabled and (len(normalized) > 100 or intent != "general")
        variants = self._variants(normalized, intent) if need_rewrite else [normalized]
        return RetrievalQueryPlan(intent, need_rewrite, variants[: self.max_variants])

    def _intent(self, query: str) -> str:
        terms = set(tokenize(query))
        lowered = query.lower()
        if terms & {"overview", "summarize", "summary"} or any(
            phrase in lowered for phrase in ("key points", "main content", "main context", "main findings")
        ):
            return "summary"
        if terms & {"compare", "comparison", "difference", "differences", "versus"} or "work together" in lowered:
            return "compare"
        if terms & {"clause", "contract", "definition", "obligation", "payment", "provision", "requirement"}:
            return "clause"
        if terms & {"evidence", "grounding", "limitation", "limitations", "risk", "risks", "verify"}:
            return "verification"
        return "general"

    def _variants(self, query: str, intent: str) -> list[str]:
        suffixes = {
            "compare": ["similarities differences", "advantages disadvantages supporting evidence"],
            "summary": ["key points conclusions", "important details supporting evidence"],
            "clause": ["definitions scope conditions exceptions", "obligations requirements supporting evidence"],
            "verification": ["supporting evidence risks limitations"],
            "general": ["related evidence"],
        }
        candidates = [query, *(f"{query} {suffix}" for suffix in suffixes[intent])]
        deduplicated: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduplicated.append(normalized)
        return deduplicated
