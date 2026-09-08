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
    integration_names = {item["name"] for item in report["integrations"]}
    assert report["integration_count"] == 6
    assert report["integrations"][0]["name"] == "local_hybrid_store"
    assert "llm_answer" in integration_names


def test_cli_integrations_can_probe_services(monkeypatch, capsys) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def read(self):
            return b'{"models":[{"name":"qwen2.5:3b"}]}'

    monkeypatch.setattr("multi_agent_rag.integrations.urlrequest.urlopen", lambda _url, timeout: FakeResponse())

    exit_code = main(["integrations", "--probe-services", "--json"])

    report = json.loads(capsys.readouterr().out)
    statuses = {item["name"]: item for item in report["integrations"]}
    assert exit_code == 0
    assert statuses["llm_answer"]["status"] == "ready"
