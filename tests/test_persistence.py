from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from bson import ObjectId

from multi_agent_rag.persistence import ConversationRepository, DocumentRepository, DocumentStatus, MongoSettings


def test_mongo_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("MONGODB_URI", "mongodb://database:27017")
    monkeypatch.setenv("MONGODB_DATABASE", "rag_test")
    monkeypatch.setenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "1500")

    settings = MongoSettings.from_env()

    assert settings.uri == "mongodb://database:27017"
    assert settings.database == "rag_test"
    assert settings.server_selection_timeout_ms == 1500


def test_document_repository_creates_processing_record() -> None:
    collection = MagicMock()
    document_id = ObjectId()
    collection.insert_one.return_value = SimpleNamespace(inserted_id=document_id)
    repository = DocumentRepository(collection)

    record = repository.create(
        title="Research Notes",
        file_type="pdf",
        file_path="output/uploads/research.pdf",
        file_size=1200,
        file_hash="abc123",
    )

    assert record.document_id == str(document_id)
    assert record.status is DocumentStatus.PROCESSING
    assert record.progress_percentage == 0
    payload = collection.insert_one.call_args.args[0]
    assert payload["current_stage"] == "upload"


def test_document_repository_updates_progress_and_completion() -> None:
    collection = MagicMock()
    collection.update_one.return_value = SimpleNamespace(matched_count=1)
    repository = DocumentRepository(collection)
    document_id = str(ObjectId())

    assert repository.update_progress(document_id, 60, "embedding", "12 of 20 chunks")
    assert repository.update_status(document_id, DocumentStatus.COMPLETED)

    progress_update = collection.update_one.call_args_list[0].args[1]["$set"]
    completed_update = collection.update_one.call_args_list[1].args[1]["$set"]
    assert progress_update["progress_percentage"] == 60
    assert progress_update["current_stage"] == "embedding"
    assert completed_update["progress_percentage"] == 100
    assert completed_update["status"] == "completed"


def test_document_repository_reads_existing_record() -> None:
    collection = MagicMock()
    document_id = ObjectId()
    now = datetime.now(UTC)
    collection.find_one.return_value = {
        "_id": document_id,
        "title": "Guide",
        "file_type": "md",
        "file_path": "guide.md",
        "file_size": 50,
        "file_hash": "hash",
        "status": "completed",
        "progress_percentage": 100,
        "current_stage": "completed",
        "stage_details": "",
        "created_at": now,
        "updated_at": now,
        "metadata": {"source": "test"},
    }

    record = DocumentRepository(collection).get(str(document_id))

    assert record is not None
    assert record.status is DocumentStatus.COMPLETED
    assert record.metadata == {"source": "test"}


def test_conversation_repository_creates_and_adds_messages() -> None:
    collection = MagicMock()
    collection.update_one.return_value = SimpleNamespace(matched_count=1)
    repository = ConversationRepository(collection)

    conversation = repository.create("Document review")
    message = repository.add_message(conversation.conversation_id, role="user", content="Summarize the document.")

    assert conversation.title == "Document review"
    assert conversation.messages == []
    assert message.role == "user"
    assert message.content == "Summarize the document."
    update = collection.update_one.call_args.args[1]
    assert update["$push"]["messages"]["message_id"] == message.message_id


def test_conversation_repository_rejects_invalid_message() -> None:
    repository = ConversationRepository(MagicMock())

    try:
        repository.add_message("conversation-id", role="system", content="Hidden instructions")
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("Expected an invalid role to be rejected.")
