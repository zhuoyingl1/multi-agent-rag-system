"""Adaptive result selection for reranked retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

from multi_agent_rag.models import SearchResult


@dataclass(frozen=True)
class AdaptiveSelection:
    """Selected evidence and diagnostics for one retrieval request."""

    results: list[SearchResult]
    selected_k: int
    estimated_context_tokens: int
    reason: str


class AdaptiveTopKPolicy:
    """Choose evidence count from reranker score separation and context budget."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_results: int = 3,
        max_results: int = 8,
        high_gap: float = 2.0,
        low_gap: float = 0.6,
        context_budget_tokens: int = 1600,
    ) -> None:
        self.enabled = enabled
        self.min_results = max(1, min_results)
        self.max_results = max(self.min_results, max_results)
        self.high_gap = high_gap
        self.low_gap = low_gap
        self.context_budget_tokens = max(1, context_budget_tokens)

    def select(self, ranked: list[SearchResult], default_k: int) -> AdaptiveSelection:
        if not ranked:
            return AdaptiveSelection([], 0, 0, "no_candidates")

        minimum = min(self.min_results, len(ranked))
        maximum = min(self.max_results, len(ranked))
        target = min(max(default_k, minimum), maximum)
        reason = "default"
        if self.enabled and len(ranked) > target:
            comparison_score = ranked[target - 1].score
            score_gap = ranked[0].score - comparison_score
            if score_gap >= self.high_gap:
                target = minimum
                reason = "concentrated_scores"
            elif score_gap <= self.low_gap:
                target = maximum
                reason = "similar_scores"

        selected: list[SearchResult] = []
        estimated_tokens = 0
        for result in ranked[:target]:
            result_tokens = estimate_tokens(result.chunk.text)
            if selected and estimated_tokens + result_tokens > self.context_budget_tokens:
                reason = "context_budget"
                break
            selected.append(result)
            estimated_tokens += result_tokens

        return AdaptiveSelection(selected, len(selected), estimated_tokens, reason)


def estimate_tokens(text: str) -> int:
    """Estimate model tokens conservatively without a tokenizer dependency."""

    return max(1, math.ceil(len(text) / 4))
