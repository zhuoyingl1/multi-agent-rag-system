import json

import pytest

from multi_agent_rag.evaluation_judge import (
    DeterministicEvaluationJudge,
    FallbackEvaluationJudge,
    OllamaEvaluationJudge,
    create_evaluation_judge,
)
from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult


def source() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            document_id="doc-1",
            chunk_id="chunk-1",
            text="RAG grounds answers in retrieved source evidence.",
            chunk_type=ChunkType.PROSE,
            index=0,
            metadata={"title": "RAG notes"},
        ),
        score=1.0,
        retrieval_type=RetrievalType.HYBRID,
    )


def test_deterministic_evaluation_judge_scores_expected_term_coverage() -> None:
    judgment = DeterministicEvaluationJudge().judge(
        "How does RAG reduce hallucination?",
        "RAG uses source evidence.",
        [source()],
        0.8,
        ["RAG", "source evidence"],
    )

    assert judgment.provider == "deterministic"
    assert judgment.grounding_score == 0.8
    assert judgment.relevance_score == 1.0
    assert judgment.completeness_score == 1.0


def test_ollama_evaluation_judge_parses_structured_scores(monkeypatch) -> None:
    judge = OllamaEvaluationJudge("qwen2.5:3b")
    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "message": {
                "content": json.dumps(
                    {
                        "grounding_score": 0.9,
                        "relevance_score": 0.8,
                        "completeness_score": 0.7,
                        "reason": "The answer is supported by the evidence.",
                        "unsupported_claims": [],
                    }
                )
            }
        }

    monkeypatch.setattr(judge, "_post_json", fake_post)

    judgment = judge.judge("What is RAG?", "RAG uses evidence.", [source()], 0.0, [])

    assert captured["path"] == "/api/chat"
    assert captured["payload"]["format"] == "json"
    assert judgment.provider == "llm"
    assert judgment.model == "qwen2.5:3b"
    assert judgment.grounding_score == 0.9


def test_evaluation_judge_falls_back_and_records_provider_error(monkeypatch) -> None:
    primary = OllamaEvaluationJudge("qwen2.5:3b")
    monkeypatch.setattr(primary, "_post_json", lambda *_args: (_ for _ in ()).throw(TimeoutError("slow model")))

    judgment = FallbackEvaluationJudge(primary).judge(
        "What is RAG?",
        "RAG uses evidence.",
        [source()],
        1.0,
        ["RAG"],
    )

    assert judgment.provider == "deterministic_fallback"
    assert judgment.model == "qwen2.5:3b"
    assert "TimeoutError" in judgment.fallback_reason


def test_required_evaluation_judge_raises_when_ollama_fails(monkeypatch) -> None:
    primary = OllamaEvaluationJudge("qwen2.5:3b")
    monkeypatch.setattr(primary, "_post_json", lambda *_args: (_ for _ in ()).throw(TimeoutError("slow model")))

    with pytest.raises(RuntimeError, match="LLM judge unavailable"):
        FallbackEvaluationJudge(primary, required=True).judge(
            "What is RAG?",
            "RAG uses evidence.",
            [source()],
            1.0,
            ["RAG"],
        )


def test_create_evaluation_judge_supports_ollama(monkeypatch) -> None:
    monkeypatch.setenv("EVALUATION_JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("EVALUATION_JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("EVALUATION_JUDGE_TIMEOUT_SECONDS", "12")

    judge = create_evaluation_judge()

    assert isinstance(judge, FallbackEvaluationJudge)
    assert judge.primary.model == "judge-model"
    assert judge.primary.timeout_seconds == 12
