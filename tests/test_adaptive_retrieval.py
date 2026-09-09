from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.retrieval.adaptive import AdaptiveTopKPolicy, estimate_tokens


def result(index: int, score: float, text: str | None = None) -> SearchResult:
    return SearchResult(
        chunk=Chunk("doc-1", f"chunk-{index}", text or f"Evidence chunk {index}.", ChunkType.PROSE, index),
        score=score,
        retrieval_type=RetrievalType.RERANKED,
    )


def test_concentrated_scores_reduce_result_count() -> None:
    ranked = [result(index, score) for index, score in enumerate([4.0, 3.7, 3.4, 2.8, 1.0, 0.8])]
    policy = AdaptiveTopKPolicy(min_results=3, max_results=8, high_gap=2.0, low_gap=0.6)

    selection = policy.select(ranked, default_k=5)

    assert selection.selected_k == 3
    assert selection.reason == "concentrated_scores"


def test_similar_scores_expand_result_count() -> None:
    ranked = [result(index, 1.0 - index * 0.04) for index in range(8)]
    policy = AdaptiveTopKPolicy(min_results=3, max_results=8, high_gap=2.0, low_gap=0.6)

    selection = policy.select(ranked, default_k=5)

    assert selection.selected_k == 8
    assert selection.reason == "similar_scores"


def test_context_budget_limits_selected_results() -> None:
    text = "A" * 80
    ranked = [result(index, 1.0 - index * 0.01, text) for index in range(8)]
    policy = AdaptiveTopKPolicy(max_results=8, context_budget_tokens=30)

    selection = policy.select(ranked, default_k=5)

    assert selection.selected_k == 1
    assert selection.estimated_context_tokens == estimate_tokens(text)
    assert selection.reason == "context_budget"


def test_disabled_policy_keeps_default_result_count() -> None:
    ranked = [result(index, 1.0 - index * 0.04) for index in range(8)]
    policy = AdaptiveTopKPolicy(enabled=False, min_results=3, max_results=8)

    selection = policy.select(ranked, default_k=5)

    assert selection.selected_k == 5
    assert selection.reason == "default"
