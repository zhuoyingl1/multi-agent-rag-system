from multi_agent_rag.agents.experts import ExpertAgent
from multi_agent_rag.agents.summarizer import SummarizerAgent
from multi_agent_rag.models import Document, JudgeResult
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


def test_summarizer_outputs_structured_sections() -> None:
    workflow = MultiAgentRAGWorkflow(HybridRetriever())
    document = Document(title="rag.md", text="RAG reduces hallucination by grounding answers in source evidence.")
    workflow.retriever.index(chunk_document(document))

    result = workflow.run("How does RAG reduce hallucination?")

    assert "Answer:" in result.answer
    assert "Evidence:" in result.answer
    assert "Unsupported claims:" in result.answer
    assert "Sources:" in result.answer


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

    assert "Answer:\nThe strongest retrieved match is Multi-Agent RAG System" in result.answer
    assert "supported by evidence mentioning" in result.answer
    assert "Analysis:" in result.answer


def test_summarizer_lists_unsupported_claims() -> None:
    answer = SummarizerAgent().summarize(
        query="What is unsupported?",
        agent_results=[],
        judge=JudgeResult(score=0.0, reason="No support", unsupported_claims=["No evidence"]),
        sources=[],
    )

    assert "Unsupported claims: No evidence" in answer
