from pathlib import Path

from multi_agent_rag.cli import main


def test_cli_ask_runs_demo(tmp_path, capsys) -> None:
    document = tmp_path / "demo.md"
    document.write_text("RAG grounds answers in source evidence and reduces hallucination risk.", encoding="utf-8")

    exit_code = main(["ask", "How does RAG reduce hallucination?", "--document", str(document), "--orchestrator", "local"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Grounding score" in output
    assert "Agents:" in output
    assert "Metrics:" in output
