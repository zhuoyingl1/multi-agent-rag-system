import pytest

from multi_agent_rag.models import Document
from multi_agent_rag.orchestration import create_workflow
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow


def build_retriever() -> HybridRetriever:
    document = Document(title="rag.md", text="RAG uses retrieved evidence to reduce hallucination.")
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    return retriever


def test_create_workflow_uses_local_by_default() -> None:
    workflow = create_workflow(build_retriever())

    assert isinstance(workflow, MultiAgentRAGWorkflow)


def test_create_workflow_auto_falls_back_without_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_LANGGRAPH", "true")
    monkeypatch.setattr("multi_agent_rag.orchestration._langgraph_available", lambda: False)

    workflow = create_workflow(build_retriever())

    assert isinstance(workflow, MultiAgentRAGWorkflow)


def test_create_workflow_rejects_forced_langgraph_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multi_agent_rag.orchestration._langgraph_available", lambda: False)

    with pytest.raises(RuntimeError, match="LangGraph orchestrator requested"):
        create_workflow(build_retriever(), orchestrator="langgraph")
