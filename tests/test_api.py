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


def test_query_endpoint_returns_grounded_answer() -> None:
    client = TestClient(build_app())

    response = client.post("/query", json={"query": "How does RAG reduce hallucination?"})

    assert response.status_code == 200
    data = response.json()
    assert "Answer:" in data["answer"]
    assert data["metrics"]["retrieved_sources"] >= 1


def test_metrics_endpoint_updates_after_query() -> None:
    client = TestClient(build_app())

    client.post("/query", json={"query": "How does RAG reduce hallucination?"})
    response = client.get("/health/metrics")

    assert response.status_code == 200
    assert response.json()["run_count"] >= 1
