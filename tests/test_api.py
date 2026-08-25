import pytest

pytest.importorskip("fastapi")
pytest.importorskip("anyio")

from fastapi.testclient import TestClient

from multi_agent_rag.api.main import build_app


def test_health_endpoint() -> None:
    client = TestClient(build_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_cors_allows_local_frontend() -> None:
    client = TestClient(build_app())

    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_query_endpoint_returns_grounded_answer() -> None:
    client = TestClient(build_app())

    response = client.post("/query", json={"query": "How does RAG reduce hallucination?"})

    assert response.status_code == 200
    data = response.json()
    assert "Answer:" in data["answer"]
    assert data["metrics"]["retrieved_sources"] >= 1


def test_query_endpoint_returns_fallback_for_insufficient_evidence() -> None:
    client = TestClient(build_app())

    response = client.post("/query", json={"query": "Who won the 1998 world chess championship?"})

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert data["metrics"]["evidence_status"] == "insufficient"
    assert "No sufficiently relevant retrieved evidence" in data["answer"]


def test_stream_endpoint_returns_all_events() -> None:
    client = TestClient(build_app())

    response = client.post("/query/stream", json={"query": "How does RAG reduce hallucination?"})

    assert response.status_code == 200
    body = response.text
    assert "event: planning" in body
    assert "event: retrieval" in body
    assert "event: agents" in body
    assert "event: judge" in body
    assert "event: answer_delta" in body
    assert "event: final" in body
    assert body.index("event: answer_delta") < body.index("event: final")


def test_metrics_endpoint_updates_after_query() -> None:
    client = TestClient(build_app())

    client.post("/query", json={"query": "How does RAG reduce hallucination?"})
    response = client.get("/health/metrics")

    assert response.status_code == 200
    assert response.json()["run_count"] >= 1


def test_integrations_endpoint_returns_readiness() -> None:
    client = TestClient(build_app())

    response = client.get("/health/integrations")

    assert response.status_code == 200
    data = response.json()
    assert data["integration_count"] == 5
    assert data["integrations"][0]["name"] == "local_hybrid_store"
    assert data["integrations"][0]["status"] == "ready"


def test_evaluate_endpoint_returns_report() -> None:
    client = TestClient(build_app())

    response = client.post("/evaluate", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["case_count"] == 3
    assert data["passed_count"] == 3
    assert data["failed_count"] == 0
