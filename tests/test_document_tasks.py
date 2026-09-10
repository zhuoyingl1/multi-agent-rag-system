from types import SimpleNamespace
from unittest.mock import MagicMock

from multi_agent_rag.document_tasks import process_document
from multi_agent_rag.persistence import DocumentStatus

def test_document_task_runs_shared_ingestion_pipeline(monkeypatch) -> None:
    service = MagicMock()
    service.process.return_value = 4
    service.get.return_value = SimpleNamespace(status=DocumentStatus.COMPLETED)
    monkeypatch.setattr("multi_agent_rag.document_tasks.create_document_ingestion_service", lambda _store: service)

    result = process_document.run("507f1f77bcf86cd799439011", "output/uploads/uploaded.md")

    assert result == {"document_id": "507f1f77bcf86cd799439011", "chunk_count": 4}
    service.process.assert_called_once_with(
        "507f1f77bcf86cd799439011",
        "output/uploads/uploaded.md",
    )
