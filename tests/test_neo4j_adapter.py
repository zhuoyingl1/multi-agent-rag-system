from multi_agent_rag.models import Document
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.neo4j_adapter import Neo4jGraphAdapter


class FakeTx:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def run(self, statement: str, **params: object) -> list[dict[str, object]]:
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
