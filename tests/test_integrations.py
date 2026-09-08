from multi_agent_rag.integrations import IntegrationConfig, check_integrations


def test_check_integrations_reports_local_ready() -> None:
    report = check_integrations(IntegrationConfig())
    statuses = {item.name: item for item in report.integrations}

    assert report.integration_count == 6
    assert statuses["local_hybrid_store"].status == "ready"
    assert statuses["qdrant"].status in {"ready", "missing_package"}
    assert statuses["qdrant"].configured is True
    assert statuses["neo4j"].status in {"ready", "missing_package"}
    assert statuses["neo4j"].configured is True
    assert statuses["bge_reranker"].status == "missing_config"
    assert statuses["llm_answer"].status == "missing_config"
    assert statuses["langgraph"].status in {"ready", "missing_package"}
    assert statuses["langgraph"].configured is True


def test_check_integrations_reports_missing_package_when_configured() -> None:
    report = check_integrations(IntegrationConfig(qdrant_url="http://localhost:6333", qdrant_collection="documents"))
    statuses = {item.name: item for item in report.integrations}

    assert statuses["qdrant"].status in {"ready", "missing_package"}
    assert statuses["qdrant"].configured is True


def test_check_integrations_reports_local_reranker_ready() -> None:
    report = check_integrations(IntegrationConfig(reranker_model="local"))
    statuses = {item.name: item for item in report.integrations}

    assert statuses["bge_reranker"].status == "ready"
    assert statuses["bge_reranker"].required_package is None
    assert statuses["bge_reranker"].configured is True
    assert report.mode == "local_with_optional_integrations"


def test_check_integrations_reports_ollama_answer_ready() -> None:
    report = check_integrations(
        IntegrationConfig(
            llm_answer_provider="ollama",
            llm_answer_model="qwen2.5:3b",
            ollama_base_url="http://127.0.0.1:11434",
        )
    )
    statuses = {item.name: item for item in report.integrations}

    assert statuses["llm_answer"].status == "ready"
    assert statuses["llm_answer"].configured is True
    assert statuses["llm_answer"].required_package is None
    assert "Ollama" in statuses["llm_answer"].notes


def test_check_integrations_probes_ollama_model(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def read(self):
            return b'{"models":[{"name":"qwen2.5:3b"}]}'

    def fake_urlopen(url, timeout):
        assert url == "http://127.0.0.1:11434/api/tags"
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr("multi_agent_rag.integrations.urlrequest.urlopen", fake_urlopen)

    report = check_integrations(
        IntegrationConfig(
            llm_answer_provider="ollama",
            llm_answer_model="qwen2.5:3b",
            ollama_base_url="http://127.0.0.1:11434",
        ),
        probe_services=True,
    )
    statuses = {item.name: item for item in report.integrations}

    assert statuses["llm_answer"].status == "ready"
    assert "model qwen2.5:3b is available" in statuses["llm_answer"].notes


def test_check_integrations_reports_missing_ollama_model(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def read(self):
            return b'{"models":[{"name":"llama3.2:3b"}]}'

    monkeypatch.setattr("multi_agent_rag.integrations.urlrequest.urlopen", lambda _url, timeout: FakeResponse())

    report = check_integrations(
        IntegrationConfig(
            llm_answer_provider="ollama",
            llm_answer_model="qwen2.5:3b",
            ollama_base_url="http://127.0.0.1:11434",
        ),
        probe_services=True,
    )
    statuses = {item.name: item for item in report.integrations}

    assert statuses["llm_answer"].status == "missing_model"
    assert "ollama pull qwen2.5:3b" in statuses["llm_answer"].notes


def test_check_integrations_reports_unavailable_ollama(monkeypatch) -> None:
    def fail_urlopen(_url, timeout):
        raise TimeoutError("slow service")

    monkeypatch.setattr("multi_agent_rag.integrations.urlrequest.urlopen", fail_urlopen)

    report = check_integrations(
        IntegrationConfig(
            llm_answer_provider="ollama",
            llm_answer_model="qwen2.5:3b",
            ollama_base_url="http://127.0.0.1:11434",
        ),
        probe_services=True,
    )
    statuses = {item.name: item for item in report.integrations}

    assert statuses["llm_answer"].status == "unavailable"
    assert statuses["llm_answer"].package_available is False
