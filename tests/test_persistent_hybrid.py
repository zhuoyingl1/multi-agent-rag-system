from datetime import UTC, datetime
from unittest.mock import MagicMock

from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.persistence import ChunkRecord
from multi_agent_rag.retrieval.persistent_hybrid import MongoKeywordRetriever, PersistentHybridRetriever, reciprocal_rank_fusion


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk("doc-1", chunk_id, text, ChunkType.PROSE, int(chunk_id[-1]))


def result(chunk_id: str, text: str, score: float, retrieval_type: RetrievalType) -> SearchResult:
    return SearchResult(chunk(chunk_id, text), score, retrieval_type)


def test_mongo_keyword_retriever_ranks_exact_terms_with_bm25() -> None:
    repository = MagicMock()
    now = datetime.now(UTC)
    repository.list_for_document.return_value = [
        ChunkRecord("chunk-1", "doc-1", "Qdrant supports semantic vector retrieval.", "prose", 1, {}, now),
        ChunkRecord("chunk-2", "doc-1", "Payment terms define the contract fee.", "prose", 2, {}, now),
    ]
    retriever = MongoKeywordRetriever(repository, "doc-1")

    results = retriever.retrieve("What are the payment terms?")

    assert [item.chunk.chunk_id for item in results] == ["chunk-2"]
    assert results[0].retrieval_type is RetrievalType.KEYWORD
    assert {"payment", "terms"}.issubset(results[0].highlights)
    repository.list_for_document.assert_called_once_with("doc-1")


def test_reciprocal_rank_fusion_boosts_multi_signal_chunk() -> None:
    vector = [
        result("chunk-1", "Vector only", 0.8, RetrievalType.VECTOR),
        result("chunk-2", "Both signals", 0.7, RetrievalType.VECTOR),
    ]
    keyword = [
        result("chunk-2", "Both signals", 2.0, RetrievalType.KEYWORD),
        result("chunk-3", "Keyword only", 1.0, RetrievalType.KEYWORD),
    ]

    fused = reciprocal_rank_fusion([(vector, 1.0), (keyword, 0.8)], limit=3)

    assert fused[0].chunk.chunk_id == "chunk-2"
    assert fused[0].retrieval_type is RetrievalType.HYBRID
    assert fused[0].score == 2.0
    assert len({item.chunk.chunk_id for item in fused}) == 3


def test_persistent_hybrid_retriever_queries_both_sources_and_closes_resources() -> None:
    vector = MagicMock()
    keyword = MagicMock()
    store = MagicMock()
    shared = result("chunk-2", "Both signals", 0.7, RetrievalType.VECTOR)
    vector.retrieve.return_value = [result("chunk-1", "Vector only", 0.8, RetrievalType.VECTOR), shared]
    keyword.retrieve.return_value = [result("chunk-2", "Both signals", 2.0, RetrievalType.KEYWORD)]
    retriever = PersistentHybridRetriever(vector, keyword, store)

    results = retriever.retrieve("hybrid retrieval", top_k=2)
    retriever.close()

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-2"
    assert retriever.last_candidate_count == 2
    vector.retrieve.assert_called_once_with("hybrid retrieval", 50)
    keyword.retrieve.assert_called_once_with("hybrid retrieval", 50)
    vector.close.assert_called_once()
    store.close.assert_called_once()
