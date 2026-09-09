from types import SimpleNamespace

import pytest

from multi_agent_rag.models import Document, RetrievalType
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.vector_index import QdrantDocumentIndex


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0, 0.5] for text in texts]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.created_vector_size: int | None = None
        self.deleted: list[object] = []
        self.upsert_batches: list[list[dict[str, object]]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.collections.add(collection_name)
        if isinstance(vectors_config, dict):
            self.created_vector_size = int(vectors_config["size"])
        else:
            self.created_vector_size = int(vectors_config.size)

    def delete(self, collection_name: str, points_selector: object, wait: bool) -> None:
        self.deleted.append(points_selector)

    def upsert(self, collection_name: str, points: list[dict[str, object]], wait: bool) -> None:
        self.upsert_batches.append(points)

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        query_filter: object,
        limit: int,
        score_threshold: float,
        with_payload: bool,
    ) -> SimpleNamespace:
        document_id = query_filter.must[0].match.value
        points = [point for batch in self.upsert_batches for point in batch]
        matches = [point for point in points if point["payload"]["document_id"] == document_id]
        scored = [SimpleNamespace(payload=point["payload"], score=0.9) for point in matches[:limit]]
        return SimpleNamespace(points=scored)


def test_document_index_batches_embeddings_and_persists_document_payload() -> None:
    embedder = FakeEmbedder()
    client = FakeQdrantClient()
    index = QdrantDocumentIndex(
        url="http://localhost:6333",
        collection="document_chunks",
        embedder=embedder,
        batch_size=2,
        client=client,
    )
    document = Document(title="rag.md", text="# RAG\n\nFirst section.\n\nSecond section.", document_id="doc-1")
    chunks = chunk_document(document)

    indexed_count = index.replace_document_chunks("doc-1", chunks)

    assert indexed_count == len(chunks) == 3
    assert [len(batch) for batch in embedder.calls] == [2, 1]
    assert client.created_vector_size == 3
    assert [len(batch) for batch in client.upsert_batches] == [2, 1]
    points = [point for batch in client.upsert_batches for point in batch]
    assert all(point["payload"]["document_id"] == "doc-1" for point in points)
    assert all("session_id" not in point["payload"] for point in points)
    assert len({point["id"] for point in points}) == len(chunks)
    assert len(client.deleted) == 1


def test_document_index_rejects_invalid_embedding_count() -> None:
    client = FakeQdrantClient()
    embedder = SimpleNamespace(encode=lambda texts: [])
    index = QdrantDocumentIndex("http://localhost:6333", "document_chunks", embedder, client=client)
    chunks = chunk_document(Document(title="rag.md", text="Evidence", document_id="doc-1"))

    with pytest.raises(RuntimeError, match="returned 0 vectors for 1 chunks"):
        index.replace_document_chunks("doc-1", chunks)

    assert not client.upsert_batches


def test_document_index_searches_only_the_requested_document() -> None:
    embedder = FakeEmbedder()
    client = FakeQdrantClient()
    index = QdrantDocumentIndex("http://localhost:6333", "document_chunks", embedder, client=client)
    first = chunk_document(Document(title="first.md", text="RAG evidence", document_id="doc-1"))
    second = chunk_document(Document(title="second.md", text="Unrelated content", document_id="doc-2"))
    index.replace_document_chunks("doc-1", first)
    index.replace_document_chunks("doc-2", second)

    results = index.search("doc-1", "RAG evidence", limit=5)

    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-1"
    assert results[0].retrieval_type is RetrievalType.VECTOR
    assert results[0].highlights == ["rag", "evidence"]


def test_document_index_deletes_only_requested_document() -> None:
    client = FakeQdrantClient()
    client.collections.add("document_chunks")
    index = QdrantDocumentIndex(
        "http://localhost:6333",
        "document_chunks",
        FakeEmbedder(),
        client=client,
    )

    index.delete_document("doc-1")

    assert len(client.deleted) == 1
