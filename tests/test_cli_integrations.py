import json

from multi_agent_rag.cli import main


def test_cli_integrations_prints_readiness(capsys) -> None:
    exit_code = main(["integrations"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Integration readiness:" in output
    assert "local_hybrid_store: ready" in output
    assert "qdrant:" in output


def test_cli_integrations_json_output(capsys) -> None:
    exit_code = main(["integrations", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["integration_count"] == 5
    assert report["integrations"][0]["name"] == "local_hybrid_store"
