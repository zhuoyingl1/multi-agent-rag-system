import pytest

from multi_agent_rag.agents.experts import ExpertAgent
from multi_agent_rag.agents.summarizer import OllamaAnswerComposer, SummarizerAgent, create_answer_composer
from multi_agent_rag.models import AgentResult, Document, JudgeResult
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow


def test_expert_selects_query_relevant_sentences() -> None:
    document = Document(
        title="resume.md",
        text=(
            "Education includes electrical engineering coursework. "
            "The Multi-Agent RAG project used FastAPI, Next.js, Qdrant, and Neo4j for document research. "
            "Other work focused on semiconductor operations."
        ),
    )
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    sources = retriever.retrieve("Which project used FastAPI and Neo4j?", top_k=3)

    result = ExpertAgent("implementation").run("Which project used FastAPI and Neo4j?", sources)

    assert "FastAPI" in result.content
    assert "Neo4j" in result.content
    assert "semiconductor operations" not in result.content.lower()


def test_summarizer_outputs_direct_answer_only() -> None:
    workflow = MultiAgentRAGWorkflow(HybridRetriever())
    document = Document(title="rag.md", text="RAG reduces hallucination by grounding answers in source evidence.")
    workflow.retriever.index(chunk_document(document))

    result = workflow.run("How does RAG reduce hallucination?")

    assert result.answer.startswith("Based on the retrieved evidence")
    assert "Question:" not in result.answer
    assert "Analysis:" not in result.answer
    assert "Evidence:" not in result.answer
    assert "Grounding score:" not in result.answer


def test_summarizer_starts_with_direct_project_answer() -> None:
    workflow = MultiAgentRAGWorkflow(HybridRetriever())
    document = Document(
        title="resume.md",
        text=(
            "Projects Multi-Agent RAG System 08/2025-12/2025 "
            "Tech Stack: LangGraph, Qdrant, Neo4j, FastAPI, Next.js. "
            "Built a multi-agent RAG platform for document research."
        ),
    )
    workflow.retriever.index(chunk_document(document))

    result = workflow.run("Which resume projects mention RAG, FastAPI, Next.js, LangGraph, Qdrant, or Neo4j?")

    assert result.answer.startswith("The strongest retrieved match is Multi-Agent RAG System")
    assert "supported by evidence mentioning" in result.answer
    assert "Analysis:" not in result.answer


def test_summarizer_lists_unsupported_claims() -> None:
    answer = SummarizerAgent().summarize(
        query="What is unsupported?",
        agent_results=[],
        judge=JudgeResult(score=0.0, reason="No support", unsupported_claims=["No evidence"]),
        sources=[],
    )

    assert "No sufficiently relevant retrieved evidence" in answer


def test_ollama_answer_composer_returns_chat_content(monkeypatch) -> None:
    document = Document(title="rag.md", text="RAG answers should use retrieved evidence and avoid unsupported claims.")
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    sources = retriever.retrieve("How should RAG answer questions?", top_k=3)
    composer = OllamaAnswerComposer(model="qwen2.5:3b", base_url="http://127.0.0.1:11434")
    captured_payload = {}

    def fake_post(path, payload):
        captured_payload["path"] = path
        captured_payload["payload"] = payload
        return {"message": {"content": "RAG should answer from retrieved evidence and avoid unsupported claims."}}

    monkeypatch.setattr(composer, "_post_json", fake_post)

    answer = composer.compose(
        "How should RAG answer questions?",
        [AgentResult("retrieval", "task", "Evidence was retrieved.", 0.9, sources=sources)],
        sources,
    )

    assert answer == "RAG should answer from retrieved evidence and avoid unsupported claims."
    assert captured_payload["path"] == "/api/chat"
    assert captured_payload["payload"]["model"] == "qwen2.5:3b"
    assert captured_payload["payload"]["stream"] is False


def test_create_answer_composer_supports_ollama(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ANSWER_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_ANSWER_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("LLM_ANSWER_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LLM_ANSWER_MAX_TOKENS", "128")

    composer = create_answer_composer()

    assert isinstance(composer, OllamaAnswerComposer)
    assert composer.model == "qwen2.5:3b"
    assert composer.base_url == "http://127.0.0.1:11434"
    assert composer.timeout_seconds == 30
    assert composer.max_tokens == 128


def test_summarizer_records_fallback_when_optional_ollama_fails(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ANSWER_REQUIRED", "false")
    document = Document(title="rag.md", text="RAG answers should use retrieved evidence.")
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    sources = retriever.retrieve("How should RAG answer?", top_k=3)
    composer = OllamaAnswerComposer(model="qwen2.5:3b")

    def fail_post(_path, _payload):
        raise TimeoutError("slow local model")

    monkeypatch.setattr(composer, "_post_json", fail_post)
    summarizer = SummarizerAgent(answer_composer=composer)

    answer = summarizer.summarize(
        query="How should RAG answer?",
        agent_results=[AgentResult("retrieval", "task", "Evidence was retrieved.", 0.9, sources=sources)],
        judge=JudgeResult(score=1.0, reason="Grounded."),
        sources=sources,
    )

    assert answer.startswith("Based on the retrieved evidence")
    assert summarizer.answer_type == "deterministic_fallback"
    assert summarizer.answer_model == "qwen2.5:3b"
    assert "slow local model" in summarizer.answer_error


def test_summarizer_raises_when_required_ollama_fails(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ANSWER_REQUIRED", "true")
    document = Document(title="rag.md", text="RAG answers should use retrieved evidence.")
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    sources = retriever.retrieve("How should RAG answer?", top_k=3)
    composer = OllamaAnswerComposer(model="qwen2.5:3b")

    def fail_post(_path, _payload):
        raise TimeoutError("slow local model")

    monkeypatch.setattr(composer, "_post_json", fail_post)
    summarizer = SummarizerAgent(answer_composer=composer)

    with pytest.raises(RuntimeError, match="LLM answer provider unavailable"):
        summarizer.summarize(
            query="How should RAG answer?",
            agent_results=[AgentResult("retrieval", "task", "Evidence was retrieved.", 0.9, sources=sources)],
            judge=JudgeResult(score=1.0, reason="Grounded."),
            sources=sources,
        )
