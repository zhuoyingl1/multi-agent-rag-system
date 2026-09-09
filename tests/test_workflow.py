from multi_agent_rag.agents.judge import GroundingJudge
from multi_agent_rag.agents.planner import PlannerAgent
from multi_agent_rag.models import Chunk, ChunkType, Document, RetrievalType, SearchResult
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.retrieval.reranking import LexicalReranker, RerankingRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow, has_enough_evidence


def build_workflow() -> MultiAgentRAGWorkflow:
    document = Document(
        title="rag.md",
        text=(
            "RAG reduces hallucination by grounding answers in source evidence. "
            "Source coverage and grounding score are useful evaluation metrics. "
            "Neo4j can expand entities across related chunks."
        ),
    )
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    return MultiAgentRAGWorkflow(retriever)


def test_planner_selects_relevant_agents() -> None:
    plan = PlannerAgent().plan("How does retrieval improve grounding metrics?")

    assert plan.selected_agents == ["retrieval", "evaluation"]
    assert "retrieval" in plan.tasks
    assert "evaluation" in plan.tasks


def test_workflow_returns_grounded_answer() -> None:
    result = build_workflow().run("How does RAG reduce hallucination with grounding metrics?")

    assert result.sources
    assert result.agents
    assert result.grounding.score > 0
    assert "Grounding score" not in result.answer
    assert result.answer.startswith("Based on the retrieved evidence")
    assert result.metrics["retrieved_sources"] >= 1
    assert result.metrics["evidence_status"] == "sufficient"
    assert result.metrics["mode"] == "deterministic_local"
    assert result.metrics["reranker"] == "none"
    assert result.metrics["query_intent"] == "general"
    assert result.metrics["query_variants"] == 1
    assert result.metrics["selected_k"] == len(result.sources)
    assert result.metrics["context_tokens"] > 0
    assert result.metrics["selection_reason"] == "fixed"
    assert result.metrics["answer_type"] == "deterministic"
    assert result.metrics["answer_model"] == "template"
    assert result.metrics["answer_error"] == ""
    assert result.metrics["citation_status"] == "missing"
    assert result.metrics["citation_coverage"] == 0.0


def test_workflow_reports_reranker_metrics() -> None:
    document = Document(
        title="rag.md",
        text="RAG reduces hallucination by grounding answers in source evidence. Latency tracks workflow speed.",
    )
    retriever = HybridRetriever()
    reranking_retriever = RerankingRetriever(retriever, LexicalReranker(), candidate_multiplier=3)
    reranking_retriever.index(chunk_document(document))

    result = MultiAgentRAGWorkflow(reranking_retriever).run("How does RAG reduce hallucination?")

    assert result.metrics["reranker"] == "lexical"
    assert result.metrics["candidate_sources"] >= result.metrics["retrieved_sources"]
    assert all(source.retrieval_type.value == "reranked" for source in result.sources)


def test_workflow_uses_fallback_for_insufficient_evidence() -> None:
    result = build_workflow().run("Who won the 1998 world chess championship?")

    assert result.sources == []
    assert result.agents == []
    assert result.grounding.score == 0.0
    assert result.metrics["retrieved_sources"] == 0
    assert result.metrics["candidate_sources"] >= 0
    assert result.metrics["evidence_status"] == "insufficient"
    assert result.metrics["answer_type"] == "deterministic"
    assert "No sufficiently relevant retrieved evidence" in result.answer


def test_semantic_vector_score_can_establish_evidence_without_exact_terms() -> None:
    source = SearchResult(
        chunk=Chunk("doc-1", "chunk-1", "Vector retrieval handles paraphrased questions.", ChunkType.PROSE, 0),
        score=0.58,
        retrieval_type=RetrievalType.VECTOR,
    )

    assert has_enough_evidence("How does the system find conceptually similar wording?", [source]) is True


def test_low_semantic_vector_score_does_not_establish_evidence() -> None:
    source = SearchResult(
        chunk=Chunk("doc-1", "chunk-1", "Unrelated content.", ChunkType.PROSE, 0),
        score=0.49,
        retrieval_type=RetrievalType.VECTOR,
    )

    assert has_enough_evidence("How does the system find conceptually similar wording?", [source]) is False


def test_judge_flags_missing_sources() -> None:
    judge = GroundingJudge().judge(agent_results=[], sources=[])

    assert judge.score == 0.0
    assert judge.unsupported_claims
