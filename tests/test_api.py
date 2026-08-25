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

    response = client.post("/query", json={"query": "How does RAG reduce hallucination?", "orchestrator": "local"})

    assert response.status_code == 200
    data = response.json()
    assert "Answer:" in data["answer"]
    assert data["metrics"]["retrieved_sources"] >= 1
    assert data["metrics"]["mode"] == "deterministic_local"


def test_query_endpoint_rejects_invalid_orchestrator() -> None:
    client = TestClient(build_app())

    response = client.post("/query", json={"query": "How does RAG reduce hallucination?", "orchestrator": "invalid"})

    assert response.status_code == 422


def test_query_endpoint_returns_fallback_for_insufficient_evidence() -> None:
    client = TestClient(build_app())

    response = client.post("/query", json={"query": "Who won the 1998 world chess championship?"})

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert data["metrics"]["evidence_status"] == "insufficient"
    assert "No sufficiently relevant retrieved evidence" in data["answer"]


def test_query_endpoint_returns_bad_request_for_missing_document() -> None:
    client = TestClient(build_app())

    response = client.post("/query", json={"query": "What is this?", "document_path": "missing.md"})

    assert response.status_code == 400
    assert "Document not found" in response.json()["detail"]


def test_query_endpoint_rejects_empty_fields() -> None:
    client = TestClient(build_app())

    response = client.post("/query", json={"query": "", "document_path": ""})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert len(errors) == 2
    assert {tuple(error["loc"]) for error in errors} == {("body", "query"), ("body", "document_path")}


def test_upload_document_returns_queryable_path() -> None:
    client = TestClient(build_app())

    upload = client.post(
        "/documents/upload",
        files={"file": ("uploaded.md", b"# Uploaded Notes\n\nRAG uses retrieved evidence to reduce hallucination.", "text/markdown")},
    )

    assert upload.status_code == 200
    uploaded = upload.json()
    assert uploaded["filename"] == "uploaded.md"
    assert uploaded["document_path"].endswith(".md")
    assert uploaded["size_bytes"] > 0

    response = client.post(
        "/query",
        json={"query": "How does RAG reduce hallucination?", "document_path": uploaded["document_path"]},
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["retrieved_sources"] >= 1


def test_upload_document_rejects_unsupported_extension() -> None:
    client = TestClient(build_app())

    response = client.post(
        "/documents/upload",
        files={"file": ("notes.exe", b"not a supported document", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported document extension" in response.json()["detail"]


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


def test_stream_endpoint_returns_bad_request_for_missing_document() -> None:
    client = TestClient(build_app())

    response = client.post("/query/stream", json={"query": "What is this?", "document_path": "missing.md"})

    assert response.status_code == 400
    assert "Document not found" in response.json()["detail"]


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


def test_evaluate_endpoint_returns_bad_request_for_missing_cases() -> None:
    client = TestClient(build_app())

    response = client.post("/evaluate", json={"cases_path": "missing-cases.json"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Evaluation cases not found: missing-cases.json"


def test_evaluate_endpoint_rejects_empty_cases_path() -> None:
    client = TestClient(build_app())

    response = client.post("/evaluate", json={"cases_path": ""})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "cases_path"]
