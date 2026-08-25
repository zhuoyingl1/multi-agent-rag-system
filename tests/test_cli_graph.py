from multi_agent_rag.cli import main


class FakeGraphAdapter:
    def __init__(self, uri: str, user: str, password: str, database: str) -> None:
        self.indexed = False

    def index(self, chunks) -> None:
        self.indexed = True

    def expand_entities(self, query: str) -> list[str]:
        return ["qdrant", "langgraph"]

    def close(self) -> None:
        return None


def test_cli_graph_indexes_document(monkeypatch, tmp_path, capsys) -> None:
    document = tmp_path / "graph.md"
    document.write_text("Neo4j connects RAG chunks with Qdrant and LangGraph.", encoding="utf-8")
    monkeypatch.setattr("multi_agent_rag.cli.Neo4jGraphAdapter", FakeGraphAdapter)

    exit_code = main(["graph", str(document), "--query", "How does Neo4j help RAG?"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Chunks indexed:" in output
    assert "qdrant" in output
    assert "langgraph" in output
