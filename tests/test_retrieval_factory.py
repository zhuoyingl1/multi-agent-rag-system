from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from multi_agent_rag.retrieval.factory import create_retriever
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.retrieval.qdrant_adapter import QdrantRetriever
from multi_agent_rag.retrieval.reranking import RerankingRetriever
from multi_agent_rag.runtime_config import RuntimeSettings


@pytest.fixture
def qdrant_client_stub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    monkeypatch.setattr(QdrantRetriever, "_build_client", lambda self, url: client)
    return client


def test_create_retriever_uses_external_qdrant_by_default(
    monkeypatch: pytest.MonkeyPatch,
    qdrant_client_stub: MagicMock,
) -> None:
    pytest.importorskip("qdrant_client")
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    retriever = create_retriever()

    assert isinstance(retriever, QdrantRetriever)
    assert retriever.url == "http://localhost:6333"
    assert retriever.client is qdrant_client_stub


def test_create_retriever_uses_local_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    retriever = create_retriever("local")

    assert isinstance(retriever, HybridRetriever)


def test_create_retriever_uses_env_qdrant_config(
    monkeypatch: pytest.MonkeyPatch,
    qdrant_client_stub: MagicMock,
) -> None:
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "research")
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    retriever = create_retriever("qdrant")

    assert isinstance(retriever, QdrantRetriever)
    assert retriever.url == "http://localhost:6333"
    assert retriever.collection == "research"
    assert retriever.client is qdrant_client_stub


def test_create_retriever_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Retrieval backend"):
        create_retriever("unknown")


def test_create_retriever_wraps_backend_when_reranker_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "local")
    monkeypatch.setenv("RERANKER_CANDIDATE_MULTIPLIER", "4")

    retriever = create_retriever("local")

    assert isinstance(retriever, RerankingRetriever)
    assert retriever.last_reranker == "lexical"
    assert retriever.candidate_multiplier == 4


def test_create_retriever_applies_explicit_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "local")
    settings = replace(
        RuntimeSettings(),
        top_k=2,
        dynamic_top_k_max=6,
        context_budget_tokens=900,
        reranker_candidate_multiplier=5,
    )

    retriever = create_retriever("local", runtime_settings=settings)

    assert isinstance(retriever, RerankingRetriever)
    assert isinstance(retriever.retriever, HybridRetriever)
    assert retriever.retriever.top_k == 2
    assert retriever.candidate_multiplier == 5
    assert retriever.selection_policy is not None
    assert retriever.selection_policy.max_results == 6
    assert retriever.selection_policy.context_budget_tokens == 900
