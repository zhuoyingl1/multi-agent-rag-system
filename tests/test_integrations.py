from multi_agent_rag.integrations import IntegrationConfig, check_integrations


def test_check_integrations_reports_local_ready() -> None:
    report = check_integrations(IntegrationConfig())
    statuses = {item.name: item for item in report.integrations}

    assert report.integration_count == 5
    assert statuses["local_hybrid_store"].status == "ready"
    assert statuses["qdrant"].status == "missing_config"
    assert statuses["neo4j"].status == "missing_config"
    assert statuses["bge_reranker"].status == "missing_config"
    assert statuses["langgraph"].status in {"ready", "missing_package"}
    assert statuses["langgraph"].configured is True


def test_check_integrations_reports_missing_package_when_configured() -> None:
    report = check_integrations(IntegrationConfig(qdrant_url="http://localhost:6333", qdrant_collection="documents"))
    statuses = {item.name: item for item in report.integrations}

    assert statuses["qdrant"].status in {"ready", "missing_package"}
    assert statuses["qdrant"].configured is True
