from datetime import UTC, datetime
from unittest.mock import MagicMock

from multi_agent_rag.models import Document, RetrievalType
from multi_agent_rag.persistence import ChunkRecord
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.neo4j_adapter import GraphChunkMatch, Neo4jGraphAdapter, Neo4jGraphRetriever


class FakeTx:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []
        self.last_params: dict[str, object] = {}

    def run(self, statement: str, **params: object) -> list[dict[str, object]]:
        if "RETURN candidate.id AS chunk_id" in statement:
            self.last_params = params
            return [{"chunk_id": "chunk-1", "score": 2.0, "entities": ["neo4j", "rag"]}]
        if "RETURN related.name AS entity" in statement:
            return [{"entity": "qdrant"}, {"entity": "langgraph"}]
        self.writes.append(params)
        return []


class FakeSession:
    def __init__(self, tx: FakeTx) -> None:
        self.tx = tx

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def run(self, statement: str) -> None:
        return None

    def execute_write(self, callback, *args):
        return callback(self.tx, *args)

    def execute_read(self, callback, *args):
        return callback(self.tx, *args)


class FakeDriver:
    def __init__(self) -> None:
        self.tx = FakeTx()
        self.closed = False

    def session(self, database: str) -> FakeSession:
        return FakeSession(self.tx)

    def close(self) -> None:
        self.closed = True


def test_neo4j_adapter_indexes_chunks_and_expands_entities() -> None:
    driver = FakeDriver()
    adapter = Neo4jGraphAdapter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123",
        driver=driver,
    )
    chunks = chunk_document(Document(title="graph.md", text="Neo4j connects RAG chunks with Qdrant and LangGraph."))

    adapter.index(chunks)
    related = adapter.expand_entities("How does Neo4j help RAG?")
    adapter.close()

    assert driver.tx.writes
    assert "neo4j" in driver.tx.writes[0]["entities"]
    assert related == ["qdrant", "langgraph"]
    assert driver.closed is True


def test_neo4j_adapter_replaces_document_graph_before_indexing() -> None:
    driver = FakeDriver()
    adapter = Neo4jGraphAdapter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123",
        driver=driver,
    )
    chunks = chunk_document(Document(title="graph.md", text="Neo4j connects current RAG evidence."))

    count = adapter.replace_document_chunks(chunks[0].document_id, chunks)

    assert count == len(chunks)
    assert driver.tx.writes[0] == {"document_id": chunks[0].document_id}
    assert driver.tx.writes[1] == {"document_id": chunks[0].document_id}
    assert driver.tx.writes[2]["chunk_id"] == chunks[0].chunk_id
    assert driver.tx.writes[-1] == {}


def test_neo4j_adapter_deletes_document_graph() -> None:
    driver = FakeDriver()
    adapter = Neo4jGraphAdapter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123",
        driver=driver,
    )

    adapter.delete_document("document-id")

    assert driver.tx.writes == [
        {"document_id": "document-id"},
        {"document_id": "document-id"},
        {},
    ]


def test_neo4j_adapter_retrieves_document_scoped_chunk_matches() -> None:
    driver = FakeDriver()
    adapter = Neo4jGraphAdapter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123",
        driver=driver,
    )

    matches = adapter.retrieve_chunk_matches("how does neo4j support rag?", "doc-1", limit=10)

    assert matches[0].chunk_id == "chunk-1"
    assert matches[0].score == 2.0
    assert matches[0].entities == ["neo4j", "rag"]
    assert {"neo4j", "rag"}.issubset(driver.tx.last_params["entities"])


def test_graph_retriever_returns_persisted_chunk_content() -> None:
    adapter = MagicMock()
    adapter.retrieve_chunk_matches.return_value = [
        GraphChunkMatch("chunk-1", 2.0, ["neo4j", "rag"])
    ]
    chunks = MagicMock()
    chunks.list_for_document.return_value = [
        ChunkRecord(
            "chunk-1",
            "doc-1",
            "Neo4j connects related RAG evidence.",
            "prose",
            1,
            {"title": "graph.md"},
            datetime.now(UTC),
        )
    ]
    retriever = Neo4jGraphRetriever(adapter, chunks, "doc-1")

    results = retriever.retrieve("How does Neo4j support RAG?")
    retriever.close()

    assert results[0].chunk.text == "Neo4j connects related RAG evidence."
    assert results[0].retrieval_type is RetrievalType.GRAPH
    assert results[0].highlights == ["neo4j", "rag"]
    chunks.list_for_document.assert_called_once_with("doc-1")
    adapter.close.assert_called_once()
