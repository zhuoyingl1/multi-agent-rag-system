from datetime import UTC, datetime
from unittest.mock import MagicMock

from multi_agent_rag.ingestion import CHUNKING_VERSION, INDEX_VERSION, DocumentIngestionService, document_index_is_stale
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
    documents.complete_indexing.assert_called_once_with(
        "document-id",
        index_version=INDEX_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model="none",
        chunk_count=2,
    )


def test_ingestion_persists_chunks_in_mongodb_and_qdrant(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nRAG grounds answers in retrieved evidence.", encoding="utf-8")
    documents = MagicMock()
    chunks = MagicMock()
    vector_index = MagicMock()
    vector_index.embedder.model_name = "nomic-embed-text"
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
    graph_index.replace_document_chunks.assert_called_once_with("document-id", stored_chunks)
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
    graph_index.replace_document_chunks.side_effect = RuntimeError("Neo4j unavailable")
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


def test_prepare_reindex_marks_existing_document_as_processing(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("Evidence", encoding="utf-8")
    documents = MagicMock()
    documents.begin_indexing.return_value = True
    chunks = MagicMock()
    existing = DocumentRecord(
        document_id="document-id",
        title="notes.md",
        file_type="md",
        file_path=str(path),
        file_size=8,
        file_hash="hash",
        status=DocumentStatus.COMPLETED,
        progress_percentage=100,
        current_stage="completed",
        stage_details="",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    documents.get.side_effect = [existing, None]

    prepared = DocumentIngestionService(documents, chunks).prepare_reindex("document-id")

    documents.begin_indexing.assert_called_once_with("document-id", "Reindex requested")
    assert prepared.status is DocumentStatus.PROCESSING
    assert prepared.current_stage == "queued"


def test_prepare_reindex_rejects_concurrent_indexing(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("Evidence", encoding="utf-8")
    documents = MagicMock()
    documents.get.return_value = DocumentRecord(
        document_id="document-id",
        title="notes.md",
        file_type="md",
        file_path=str(path),
        file_size=8,
        file_hash="hash",
        status=DocumentStatus.PROCESSING,
        progress_percentage=20,
        current_stage="parsing",
        stage_details="",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    try:
        DocumentIngestionService(documents, MagicMock()).prepare_reindex("document-id")
    except RuntimeError as exc:
        assert "already in progress" in str(exc)
    else:
        raise AssertionError("Expected concurrent reindex to be rejected.")


def test_document_index_staleness_tracks_pipeline_configuration() -> None:
    now = datetime.now(UTC)
    current = DocumentRecord(
        document_id="document-id",
        title="notes.md",
        file_type="md",
        file_path="notes.md",
        file_size=8,
        file_hash="hash",
        status=DocumentStatus.COMPLETED,
        progress_percentage=100,
        current_stage="completed",
        stage_details="",
        created_at=now,
        updated_at=now,
        index_version=INDEX_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model="nomic-embed-text",
    )

    assert document_index_is_stale(current, "nomic-embed-text") is False
    assert document_index_is_stale(current, "different-model") is True
