from multi_agent_rag.cli import main


def test_cli_ingest_summarizes_chunks(tmp_path, capsys) -> None:
    document = tmp_path / "demo.md"
    document.write_text("# Demo\n\nRAG grounds answers in source evidence.", encoding="utf-8")

    exit_code = main(["ingest", str(document)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Document: demo.md" in output
    assert "Type: text" in output
    assert "Chunks:" in output
