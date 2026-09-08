from datetime import UTC, datetime
from unittest.mock import MagicMock

from multi_agent_rag.ingestion import DocumentIngestionService
from multi_agent_rag.persistence import DocumentRecord, DocumentStatus


def test_registration_reuses_duplicate_document() -> None:
    documents = MagicMock()
    chunks = MagicMock()
    now = datetime.now(UTC)
    existing = DocumentRecord(
        document_id="507f1f77bcf86cd799439011",
        title="notes.md",
        file_type="md",
        file_path="output/uploads/notes.md",
        file_size=100,
        file_hash="existing-hash",
        status=DocumentStatus.COMPLETED,
        progress_percentage=100,
        current_stage="completed",
        stage_details="",
        created_at=now,
        updated_at=now,
    )
    documents.find_duplicate.return_value = existing
    service = DocumentIngestionService(documents, chunks)

    registered = service.register(
        title="notes.md",
        file_type="md",
        file_path="output/uploads/notes.md",
        file_size=100,
        file_hash="existing-hash",
    )

    assert registered.duplicate is True
    assert registered.document is existing
    documents.create.assert_not_called()


def test_ingestion_parses_chunks_and_completes_document(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nRAG grounds answers in retrieved evidence.", encoding="utf-8")
    documents = MagicMock()
    chunks = MagicMock()
    service = DocumentIngestionService(documents, chunks)

    chunk_count = service.process("document-id", path)

    assert chunk_count == 2
    chunks.replace_document_chunks.assert_called_once()
    stored_chunks = chunks.replace_document_chunks.call_args.args[1]
    assert all(chunk.document_id == "document-id" for chunk in stored_chunks)
    documents.update_status.assert_called_once_with("document-id", DocumentStatus.COMPLETED)


def test_ingestion_persists_chunks_in_mongodb_and_qdrant(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nRAG grounds answers in retrieved evidence.", encoding="utf-8")
    documents = MagicMock()
    chunks = MagicMock()
    vector_index = MagicMock()
    service = DocumentIngestionService(documents, chunks, vector_index)

    chunk_count = service.process("document-id", path)

    assert chunk_count == 2
    stored_chunks = chunks.replace_document_chunks.call_args.args[1]
    vector_index.replace_document_chunks.assert_called_once_with("document-id", stored_chunks)
    progress_calls = documents.update_progress.call_args_list
    assert progress_calls[-1].args[:3] == ("document-id", 85, "indexing")


def test_ingestion_indexes_chunk_relationships_in_neo4j(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nNeo4j connects RAG evidence through entities.", encoding="utf-8")
    documents = MagicMock()
    chunks = MagicMock()
    vector_index = MagicMock()
    graph_index = MagicMock()
    service = DocumentIngestionService(documents, chunks, vector_index, graph_index)

    chunk_count = service.process("document-id", path)

    assert chunk_count == 2
    stored_chunks = chunks.replace_document_chunks.call_args.args[1]
    graph_index.index.assert_called_once_with(stored_chunks)
    graph_index.close.assert_called_once()
    progress_calls = documents.update_progress.call_args_list
    assert progress_calls[-1].args[:3] == ("document-id", 92, "graph_indexing")


def test_ingestion_marks_document_failed_when_vector_indexing_fails(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("RAG grounds answers in retrieved evidence.", encoding="utf-8")
    documents = MagicMock()
    chunks = MagicMock()
    vector_index = MagicMock()
    vector_index.replace_document_chunks.side_effect = RuntimeError("Qdrant unavailable")
    service = DocumentIngestionService(documents, chunks, vector_index)

    chunk_count = service.process("document-id", path)

    assert chunk_count == 0
    status_call = documents.update_status.call_args
    assert status_call.args[1] is DocumentStatus.FAILED
    assert status_call.args[2] == "Qdrant unavailable"


def test_ingestion_marks_document_failed_when_graph_indexing_fails(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("Neo4j connects RAG evidence through entities.", encoding="utf-8")
    documents = MagicMock()
    chunks = MagicMock()
    graph_index = MagicMock()
    graph_index.index.side_effect = RuntimeError("Neo4j unavailable")
    service = DocumentIngestionService(documents, chunks, graph_index=graph_index)

    chunk_count = service.process("document-id", path)

    assert chunk_count == 0
    status_call = documents.update_status.call_args
    assert status_call.args[1] is DocumentStatus.FAILED
    assert status_call.args[2] == "Neo4j unavailable"
    graph_index.close.assert_called_once()


def test_ingestion_marks_document_failed_when_parsing_fails(tmp_path) -> None:
    documents = MagicMock()
    chunks = MagicMock()
    service = DocumentIngestionService(documents, chunks)

    chunk_count = service.process("document-id", tmp_path / "missing.md")

    assert chunk_count == 0
    status_call = documents.update_status.call_args
    assert status_call.args[1] is DocumentStatus.FAILED
    assert "Document not found" in status_call.args[2]
    chunks.replace_document_chunks.assert_not_called()
