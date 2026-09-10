import pytest

from multi_agent_rag.cli import main


def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "Multi-Agent RAG System V2" in capsys.readouterr().out


def test_cli_plan(capsys) -> None:
    exit_code = main(["plan"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Implementation plan:" in output
    assert "Multi-agent workflow" in output


def test_cli_ingest_can_show_chunk_previews(tmp_path, capsys) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nGrounded evidence.", encoding="utf-8")

    exit_code = main(["ingest", str(document), "--show-chunks"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Chunk previews:" in output
    assert "[0] prose: # Notes" in output
