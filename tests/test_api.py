import pytest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

pytest.importorskip("fastapi")
pytest.importorskip("anyio")

from fastapi.testclient import TestClient

from multi_agent_rag.api.main import build_app, run_query
from multi_agent_rag.ingestion import RegisteredDocument
from multi_agent_rag.persistence import DocumentRecord, DocumentStatus


def fake_document(status: DocumentStatus = DocumentStatus.PROCESSING) -> DocumentRecord:
    now = datetime.now(UTC)
    return DocumentRecord(
        document_id="507f1f77bcf86cd799439011",
        title="uploaded.md",
        file_type="md",
        file_path="output/uploads/uploaded.md",
        file_size=70,
        file_hash="hash",
        status=status,
        progress_percentage=100 if status is DocumentStatus.COMPLETED else 0,
        current_stage=status.value,
        stage_details="",
        created_at=now,
        updated_at=now,
    )


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


def test_cors_allows_next_fallback_port() -> None:
    client = TestClient(build_app())

    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:3001",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3001"


def test_query_endpoint_returns_grounded_answer() -> None:
    client = TestClient(build_app())

    response = client.post(
        "/query",
        json={
            "query": "How does RAG reduce hallucination?",
            "orchestrator": "local",
            "retrieval_backend": "local",
            "require_llm_answer": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("Based on the retrieved evidence")
    assert data["metrics"]["retrieved_sources"] >= 1
    assert data["metrics"]["mode"] == "deterministic_local"
    assert data["workflow_trace"]["mode"] == "deterministic_local"
    assert data["workflow_trace"]["evidence_status"] == "sufficient"
    assert data["workflow_trace"]["selected_agents"]
    assert data["workflow_trace"]["query_intent"] == "general"
    assert data["workflow_trace"]["query_variants"] == 1


def test_query_endpoint_uses_llm_answer_by_default(monkeypatch) -> None:
    def fake_post(_self, _path, _payload):
        return {"message": {"content": "RAG reduces hallucination by grounding answers in retrieved evidence."}}

    monkeypatch.setattr("multi_agent_rag.agents.summarizer.OllamaAnswerComposer._post_json", fake_post)
    client = TestClient(build_app())

    response = client.post(
        "/query",
        json={"query": "How does RAG reduce hallucination?", "orchestrator": "local", "retrieval_backend": "local"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "RAG reduces hallucination by grounding answers in retrieved evidence."
    assert data["metrics"]["answer_type"] == "llm"
    assert data["metrics"]["answer_model"] == "qwen2.5:3b"


def test_query_endpoint_uses_persistent_document_id(monkeypatch) -> None:
    expected = run_query(
        "How does RAG reduce hallucination?",
        Path("examples/sample_docs.md"),
        orchestrator="local",
        retrieval_backend="local",
        require_llm_answer=False,
    )
    captured: dict[str, str] = {}

    def fake_document_query(query, document_id, orchestrator, require_llm_answer):
        captured["query"] = query
        captured["document_id"] = document_id
        return expected

    monkeypatch.setattr("multi_agent_rag.api.main.safe_run_document_query", fake_document_query)
    client = TestClient(build_app())

    response = client.post(
        "/query",
        json={
            "query": "How does RAG reduce hallucination?",
            "document_id": "507f1f77bcf86cd799439011",
            "require_llm_answer": False,
        },
    )

    assert response.status_code == 200
    assert captured == {
        "query": "How does RAG reduce hallucination?",
        "document_id": "507f1f77bcf86cd799439011",
    }
    assert response.json()["metrics"]["retrieved_sources"] >= 1


def test_query_endpoint_rejects_invalid_orchestrator() -> None:
    client = TestClient(build_app())

    response = client.post(
        "/query",
        json={
            "query": "How does RAG reduce hallucination?",
            "orchestrator": "invalid",
            "retrieval_backend": "local",
            "require_llm_answer": False,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_invalid_retrieval_backend() -> None:
    client = TestClient(build_app())

    response = client.post(
        "/query",
        json={"query": "How does RAG reduce hallucination?", "retrieval_backend": "invalid", "require_llm_answer": False},
    )

    assert response.status_code == 422


def test_query_endpoint_returns_fallback_for_insufficient_evidence() -> None:
    client = TestClient(build_app())

    response = client.post(
        "/query",
        json={
            "query": "Who won the 1998 world chess championship?",
            "retrieval_backend": "local",
            "require_llm_answer": False,
        },
    )

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


def test_upload_document_returns_queryable_path(monkeypatch) -> None:
    service = MagicMock()
    service.register.side_effect = lambda **values: RegisteredDocument(
        replace(
            fake_document(),
            title=values["title"],
            file_path=values["file_path"],
            file_size=values["file_size"],
            file_hash=values["file_hash"],
            metadata=values["metadata"],
        ),
        duplicate=False,
    )
    monkeypatch.setattr("multi_agent_rag.api.main.create_document_ingestion_service", lambda: service)
    client = TestClient(build_app())

    upload = client.post(
        "/documents/upload",
        files={"file": ("uploaded.md", b"# Uploaded Notes\n\nRAG uses retrieved evidence to reduce hallucination.", "text/markdown")},
    )

    assert upload.status_code == 200
    uploaded = upload.json()
    assert uploaded["filename"] == "uploaded.md"
    assert uploaded["document_id"] == "507f1f77bcf86cd799439011"
    assert uploaded["document_path"].endswith(".md")
    assert uploaded["size_bytes"] > 0
    assert uploaded["status"] == "processing"
    assert uploaded["duplicate"] is False
    service.process.assert_called_once()

    response = client.post(
        "/query",
        json={
            "query": "How does RAG reduce hallucination?",
            "document_path": uploaded["document_path"],
            "retrieval_backend": "local",
            "require_llm_answer": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["retrieved_sources"] >= 1


def test_upload_document_rejects_unsupported_extension(monkeypatch) -> None:
    monkeypatch.setattr("multi_agent_rag.api.main.create_document_ingestion_service", MagicMock())
    client = TestClient(build_app())

    response = client.post(
        "/documents/upload",
        files={"file": ("notes.exe", b"not a supported document", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported document extension" in response.json()["detail"]


def test_document_status_endpoint_returns_processing_state(monkeypatch) -> None:
    repository = MagicMock()
    repository.get.return_value = fake_document()
    monkeypatch.setattr("multi_agent_rag.api.main.DocumentRepository.from_store", lambda _store: repository)
    client = TestClient(build_app())

    response = client.get("/documents/507f1f77bcf86cd799439011")

    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert response.json()["progress_percentage"] == 0


def test_stream_endpoint_returns_all_events() -> None:
    client = TestClient(build_app())

    response = client.post(
        "/query/stream",
        json={"query": "How does RAG reduce hallucination?", "retrieval_backend": "local", "require_llm_answer": False},
    )

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

    client.post(
        "/query",
        json={"query": "How does RAG reduce hallucination?", "retrieval_backend": "local", "require_llm_answer": False},
    )
    response = client.get("/health/metrics")

    assert response.status_code == 200
    assert response.json()["run_count"] >= 1


def test_integrations_endpoint_returns_readiness(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def read(self):
            return b'{"models":[{"name":"qwen2.5:3b"}]}'

    monkeypatch.setattr("multi_agent_rag.integrations.urlrequest.urlopen", lambda _url, timeout: FakeResponse())
    client = TestClient(build_app())

    response = client.get("/health/integrations")

    assert response.status_code == 200
    data = response.json()
    integration_names = {item["name"] for item in data["integrations"]}
    assert data["integration_count"] == 6
    assert data["integrations"][0]["name"] == "local_hybrid_store"
    assert data["integrations"][0]["status"] == "ready"
    assert "llm_answer" in integration_names


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
