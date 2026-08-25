from types import SimpleNamespace

from multi_agent_rag.models import Document, RetrievalType
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.embeddings import hashed_embedding
from multi_agent_rag.retrieval.qdrant_adapter import QdrantRetriever


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.points: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.collections.add(collection_name)

    def upsert(self, collection_name: str, points: list[dict[str, object]]) -> None:
        self.points.extend(points)

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        query_filter: object | None = None,
    ) -> SimpleNamespace:
        scored = []
        for point in self.points:
            vector = point["vector"]
            assert isinstance(vector, list)
            score = sum(float(left) * float(right) for left, right in zip(query, vector))
            scored.append(SimpleNamespace(payload=point["payload"], score=round(score, 4)))
        return SimpleNamespace(points=sorted(scored, key=lambda item: item.score, reverse=True)[:limit])


def test_hashed_embedding_is_stable_and_normalized() -> None:
    first = hashed_embedding("RAG retrieved evidence")
    second = hashed_embedding("RAG retrieved evidence")

    assert first == second
    assert len(first) == 64
    assert any(value != 0 for value in first)


def test_qdrant_retriever_indexes_and_retrieves_chunks() -> None:
    client = FakeQdrantClient()
    retriever = QdrantRetriever(url="http://localhost:6333", collection="documents", client=client)
    document = Document(title="rag.md", text="RAG reduces hallucination with retrieved evidence.")
    retriever.index(chunk_document(document))

    results = retriever.retrieve("retrieved evidence", top_k=1)

    assert results
    assert results[0].retrieval_type is RetrievalType.VECTOR
    assert "retrieved" in results[0].highlights
    assert "evidence" in results[0].chunk.text.lower()


def test_qdrant_point_ids_are_session_scoped() -> None:
    client = FakeQdrantClient()
    chunks = chunk_document(Document(title="rag.md", text="RAG reduces hallucination with retrieved evidence."))
    first = QdrantRetriever(url="http://localhost:6333", collection="documents", session_id="first", client=client)
    second = QdrantRetriever(url="http://localhost:6333", collection="documents", session_id="second", client=client)

    first.index(chunks)
    second.index(chunks)

    point_ids = [point["id"] for point in client.points]
    assert len(point_ids) == 2
    assert len(set(point_ids)) == 2
