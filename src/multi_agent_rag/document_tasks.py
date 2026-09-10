"""Celery tasks for persistent document ingestion."""

from __future__ import annotations

from multi_agent_rag.celery_app import celery_app
from multi_agent_rag.ingestion_factory import create_document_ingestion_service
from multi_agent_rag.persistence import DocumentStatus, MongoStore

WORKER_MONGO_STORE = MongoStore()


class DocumentTaskFailed(RuntimeError):
    """Raised to let Celery retry a failed ingestion task."""


@celery_app.task(
    name="multi_agent_rag.process_document",
)
def process_document(document_id: str, file_path: str) -> dict[str, int | str]:
    """Run one idempotent document ingestion job in a worker process."""

    service = create_document_ingestion_service(WORKER_MONGO_STORE)
    chunk_count = service.process(document_id, file_path)
    document = service.get(document_id)
    if document is None or document.status is DocumentStatus.FAILED:
        raise DocumentTaskFailed("Document ingestion did not complete.")
    return {"document_id": document_id, "chunk_count": chunk_count}
