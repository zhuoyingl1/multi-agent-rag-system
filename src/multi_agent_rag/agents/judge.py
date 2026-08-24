"""Grounding judge for deterministic local evaluation."""

from __future__ import annotations

from multi_agent_rag.models import AgentResult, JudgeResult, SearchResult
from multi_agent_rag.retrieval.tokenization import tokenize


class GroundingJudge:
    """Estimate whether agent findings are grounded in retrieved sources."""

    def judge(self, agent_results: list[AgentResult], sources: list[SearchResult]) -> JudgeResult:
        if not sources:
            return JudgeResult(
                score=0.0,
                reason="No retrieved sources were available, so grounding cannot be established.",
                unsupported_claims=["The answer has no retrieved evidence."],
            )

        source_tokens = set(tokenize(" ".join(source.chunk.text for source in sources)))
        draft_tokens = set(tokenize(" ".join(result.content for result in agent_results)))
        if not draft_tokens:
            return JudgeResult(score=0.0, reason="No draft content was produced.", unsupported_claims=["The agents produced no answer."])

        overlap = source_tokens & draft_tokens
        score = min(1.0, (len(overlap) / max(1, len(draft_tokens))) * 2.0)
        rounded = round(score, 2)
        unsupported = [] if rounded >= 0.35 else ["The draft has weak lexical support from retrieved sources."]
        reason = f"Grounding score is based on lexical overlap between agent findings and {len(sources)} retrieved sources."
        return JudgeResult(score=rounded, reason=reason, unsupported_claims=unsupported)
