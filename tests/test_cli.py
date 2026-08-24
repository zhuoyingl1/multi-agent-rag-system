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
