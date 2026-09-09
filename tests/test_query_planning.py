from multi_agent_rag.retrieval.query_planning import RetrievalQueryPlanner


def test_general_query_uses_only_the_original_text() -> None:
    plan = RetrievalQueryPlanner().plan("How does RAG work?")

    assert plan.intent == "general"
    assert plan.need_rewrite is False
    assert plan.query_variants == ["How does RAG work?"]


def test_comparison_query_builds_three_unique_variants() -> None:
    plan = RetrievalQueryPlanner().plan("Compare Qdrant and Neo4j retrieval.")

    assert plan.intent == "compare"
    assert plan.need_rewrite is True
    assert len(plan.query_variants) == 3
    assert plan.query_variants[0] == "Compare Qdrant and Neo4j retrieval."
    assert "similarities differences" in plan.query_variants[1]


def test_summary_and_clause_intents_add_targeted_terms() -> None:
    summary = RetrievalQueryPlanner().plan("Summarize the main content of this contract.")
    clause = RetrievalQueryPlanner().plan("What payment obligations and exceptions does the contract contain?")

    assert summary.intent == "summary"
    assert "key points conclusions" in summary.query_variants[1]
    assert clause.intent == "clause"
    assert "definitions scope conditions exceptions" in clause.query_variants[1]


def test_query_rewriting_can_be_disabled() -> None:
    plan = RetrievalQueryPlanner(rewrite_enabled=False).plan("Compare vector and graph retrieval.")

    assert plan.intent == "compare"
    assert plan.need_rewrite is False
    assert plan.query_variants == ["Compare vector and graph retrieval."]
