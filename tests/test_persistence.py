from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from bson import ObjectId

from multi_agent_rag.models import Chunk, ChunkType
from multi_agent_rag.persistence import (
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    DocumentStatus,
    KnowledgeSpaceRepository,
    MongoSettings,
)


class FakeCursor:
    def __init__(self, records):
        self.records = records

    def sort(self, *_args):
        return self

    def skip(self, value):
        self.records = self.records[value:]
        return self

    def limit(self, value):
        self.records = self.records[:value]
        return self

    def __iter__(self):
        return iter(self.records)


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


def test_knowledge_space_repository_creates_and_lists_spaces() -> None:
    collection = MagicMock()
    knowledge_space_id = ObjectId()
    collection.insert_one.return_value = SimpleNamespace(inserted_id=knowledge_space_id)
    repository = KnowledgeSpaceRepository(collection)

    created = repository.create("Research", "Related technical documents")
    collection.find.return_value = FakeCursor([collection.insert_one.call_args.args[0]])
    collection.count_documents.return_value = 1

    spaces = repository.list(limit=10)

    assert created.knowledge_space_id == str(knowledge_space_id)
    assert created.name == "Research"
    assert spaces[0].description == "Related technical documents"
    assert repository.count() == 1


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


def test_document_repository_records_index_configuration() -> None:
    collection = MagicMock()
    collection.update_one.return_value = SimpleNamespace(matched_count=1)
    repository = DocumentRepository(collection)
    document_id = str(ObjectId())

    assert repository.begin_indexing(document_id, "Reindex requested")
    assert repository.complete_indexing(
        document_id,
        index_version="1",
        chunking_version="structured-v1",
        embedding_model="nomic-embed-text",
        chunk_count=12,
    )

    started = collection.update_one.call_args_list[0].args[1]["$set"]
    started_filter = collection.update_one.call_args_list[0].args[0]
    completed = collection.update_one.call_args_list[1].args[1]["$set"]
    assert started["status"] == "processing"
    assert started["current_stage"] == "queued"
    assert started_filter["status"] == {"$ne": "processing"}
    assert completed["status"] == "completed"
    assert completed["chunk_count"] == 12
    assert completed["embedding_model"] == "nomic-embed-text"


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
    assert record.index_version is None
    assert record.chunk_count == 0


def test_document_repository_lists_counts_and_deletes_records() -> None:
    collection = MagicMock()
    document_id = ObjectId()
    now = datetime.now(UTC)
    collection.find.return_value = FakeCursor(
        [
            {
                "_id": document_id,
                "title": "Guide",
                "file_type": "md",
                "file_path": "guide.md",
                "file_size": 50,
                "file_hash": "hash",
                "status": "completed",
                "created_at": now,
                "updated_at": now,
            }
        ]
    )
    collection.count_documents.return_value = 1
    collection.delete_one.return_value = SimpleNamespace(deleted_count=1)
    repository = DocumentRepository(collection)

    records = repository.list(skip=0, limit=10, status=DocumentStatus.COMPLETED)

    assert [record.document_id for record in records] == [str(document_id)]
    assert repository.count(DocumentStatus.COMPLETED) == 1
    assert repository.delete(str(document_id)) is True
    collection.find.assert_called_once_with({"status": "completed"})
    collection.count_documents.assert_called_once_with({"status": "completed"})


def test_document_repository_filters_and_assigns_knowledge_space() -> None:
    collection = MagicMock()
    collection.find.return_value = FakeCursor([])
    collection.count_documents.return_value = 0
    collection.update_one.return_value = SimpleNamespace(matched_count=1)
    repository = DocumentRepository(collection)
    document_id = str(ObjectId())

    records = repository.list(limit=10, knowledge_space_id="space-id")
    count = repository.count(knowledge_space_id="space-id")
    assigned = repository.set_knowledge_space(document_id, "space-id")

    assert records == []
    assert count == 0
    assert assigned is True
    collection.find.assert_called_once_with({"knowledge_space_id": "space-id"})
    collection.count_documents.assert_called_once_with({"knowledge_space_id": "space-id"})
    assert collection.update_one.call_args.args[1]["$set"]["knowledge_space_id"] == "space-id"


def test_conversation_repository_creates_and_adds_messages() -> None:
    collection = MagicMock()
    collection.update_one.return_value = SimpleNamespace(matched_count=1)
    repository = ConversationRepository(collection)

    conversation = repository.create("Document review", document_id="document-id")
    message = repository.add_message(conversation.conversation_id, role="user", content="Summarize the document.")

    assert conversation.title == "Document review"
    assert conversation.messages == []
    assert conversation.document_id == "document-id"
    assert message.role == "user"
    assert message.content == "Summarize the document."
    update = collection.update_one.call_args.args[1]
    assert update["$push"]["messages"]["message_id"] == message.message_id


def test_conversation_repository_adds_turn_atomically() -> None:
    collection = MagicMock()
    collection.update_one.return_value = SimpleNamespace(matched_count=1)
    repository = ConversationRepository(collection)

    user, assistant = repository.add_turn(
        "conversation-id",
        user_content="Tell me more.",
        assistant_content="Here are more details.",
        assistant_metadata={"answer_type": "llm"},
    )

    assert user.role == "user"
    assert assistant.role == "assistant"
    assert assistant.metadata == {"answer_type": "llm"}
    update = collection.update_one.call_args.args[1]
    assert [message["role"] for message in update["$push"]["messages"]["$each"]] == ["user", "assistant"]


def test_conversation_repository_lists_scope_and_deletes() -> None:
    collection = MagicMock()
    collection.find.return_value = FakeCursor([])
    collection.delete_one.return_value = SimpleNamespace(deleted_count=1)
    repository = ConversationRepository(collection)

    conversations = repository.list(document_id="document-id", limit=20)
    deleted = repository.delete("conversation-id")

    assert conversations == []
    assert deleted is True
    collection.find.assert_called_once_with({"document_id": "document-id"})
    collection.delete_one.assert_called_once_with({"_id": "conversation-id"})


def test_conversation_repository_updates_normalized_title() -> None:
    collection = MagicMock()
    collection.update_one.return_value = SimpleNamespace(matched_count=1)

    updated = ConversationRepository(collection).update_title("conversation-id", "  Payment terms  ")

    assert updated is True
    update = collection.update_one.call_args.args[1]["$set"]
    assert update["title"] == "Payment terms"
    assert "updated_at" in update


def test_conversation_repository_rejects_empty_title() -> None:
    repository = ConversationRepository(MagicMock())

    try:
        repository.update_title("conversation-id", "   ")
    except ValueError as exc:
        assert str(exc) == "Conversation title must not be empty."
    else:
        raise AssertionError("Expected an empty conversation title to be rejected.")


def test_conversation_repository_rejects_invalid_message() -> None:
    repository = ConversationRepository(MagicMock())

    try:
        repository.add_message("conversation-id", role="system", content="Hidden instructions")
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("Expected an invalid role to be rejected.")


def test_chunk_repository_replaces_document_chunks() -> None:
    collection = MagicMock()
    chunks = [
        Chunk(
            document_id="document-id",
            chunk_id="chunk-1",
            text="Retrieved evidence",
            chunk_type=ChunkType.PROSE,
            index=0,
            metadata={"title": "Notes"},
        )
    ]

    records = ChunkRepository(collection).replace_document_chunks("document-id", chunks)

    collection.delete_many.assert_called_once_with({"document_id": "document-id"})
    collection.insert_many.assert_called_once()
    assert records[0].chunk_id == "chunk-1"
    assert records[0].chunk_type == "prose"


def test_chunk_and_conversation_repositories_delete_document_data() -> None:
    chunks = MagicMock()
    chunks.delete_many.return_value = SimpleNamespace(deleted_count=3)
    conversations = MagicMock()
    conversations.delete_many.return_value = SimpleNamespace(deleted_count=2)

    assert ChunkRepository(chunks).delete_for_document("document-id") == 3
    assert ConversationRepository(conversations).delete_for_document("document-id") == 2
    chunks.delete_many.assert_called_once_with({"document_id": "document-id"})
    conversations.delete_many.assert_called_once_with({"document_id": "document-id"})


def test_chunk_repository_lists_filtered_preview_page() -> None:
    collection = MagicMock()
    now = datetime.now(UTC)
    collection.find.return_value = FakeCursor(
        [
            {
                "_id": "chunk-2",
                "document_id": "document-id",
                "text": "Qdrant stores document vectors.",
                "chunk_type": "prose",
                "index": 2,
                "metadata": {"line_start": "8", "line_end": "9"},
                "created_at": now,
            }
        ]
    )
    collection.count_documents.return_value = 1

    records, total = ChunkRepository(collection).list_page(
        "document-id",
        skip=0,
        limit=5,
        chunk_type="prose",
        query="Qdrant?",
    )

    assert total == 1
    assert records[0].chunk_id == "chunk-2"
    expected_filter = {
        "document_id": "document-id",
        "chunk_type": "prose",
        "text": {"$regex": "Qdrant\\?", "$options": "i"},
    }
    collection.find.assert_called_once_with(expected_filter)
    collection.count_documents.assert_called_once_with(expected_filter)


def test_chunk_repository_lists_multiple_documents() -> None:
    collection = MagicMock()
    collection.find.return_value = FakeCursor([])

    records = ChunkRepository(collection).list_for_documents(["doc-1", "doc-2"])

    assert records == []
    collection.find.assert_called_once_with({"document_id": {"$in": ["doc-1", "doc-2"]}})
