"""MongoDB connection and repositories for persistent application state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from multi_agent_rag.models import Chunk
from multi_agent_rag.persistence.models import ChunkRecord, ConversationMessage, ConversationRecord, DocumentRecord, DocumentStatus


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class MongoSettings:
    """MongoDB connection settings loaded from the environment."""

    uri: str = "mongodb://localhost:27017"
    database: str = "multi_agent_rag"
    server_selection_timeout_ms: int = 3000

    @classmethod
    def from_env(cls) -> "MongoSettings":
        return cls(
            uri=os.getenv("MONGODB_URI") or "mongodb://localhost:27017",
            database=os.getenv("MONGODB_DATABASE") or "multi_agent_rag",
            server_selection_timeout_ms=int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000")),
        )


class MongoStore:
    """Own a lazy MongoDB client and expose named collections."""

    def __init__(self, settings: MongoSettings | None = None) -> None:
        self.settings = settings or MongoSettings.from_env()
        self._client: MongoClient[dict[str, Any]] | None = None
        self._database: Database[dict[str, Any]] | None = None

    def connect(self) -> Database[dict[str, Any]]:
        if self._database is None:
            self._client = MongoClient(
                self.settings.uri,
                serverSelectionTimeoutMS=self.settings.server_selection_timeout_ms,
            )
            self._client.admin.command("ping")
            self._database = self._client[self.settings.database]
        return self._database

    def collection(self, name: str) -> Collection[dict[str, Any]]:
        if not name.strip():
            raise ValueError("Collection name must not be empty.")
        return self.connect()[name]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._database = None


class DocumentRepository:
    """Persist document metadata and ingestion progress."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        self.collection = collection

    @classmethod
    def from_store(cls, store: MongoStore) -> "DocumentRepository":
        return cls(store.collection("documents"))

    def create(
        self,
        *,
        title: str,
        file_type: str,
        file_path: str,
        file_size: int,
        file_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        now = utc_now()
        payload: dict[str, Any] = {
            "title": title,
            "file_type": file_type,
            "file_path": file_path,
            "file_size": file_size,
            "file_hash": file_hash,
            "metadata": metadata or {},
            "status": DocumentStatus.PROCESSING.value,
            "progress_percentage": 0,
            "current_stage": "upload",
            "stage_details": "",
            "index_version": None,
            "chunking_version": None,
            "embedding_model": None,
            "indexed_at": None,
            "chunk_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        result = self.collection.insert_one(payload)
        payload["_id"] = result.inserted_id
        return _document_record(payload)

    def get(self, document_id: str) -> DocumentRecord | None:
        document = self.collection.find_one(_document_filter(document_id))
        return _document_record(document) if document else None

    def find_duplicate(self, file_hash: str) -> DocumentRecord | None:
        document = self.collection.find_one({"file_hash": file_hash})
        return _document_record(document) if document else None

    def update_status(self, document_id: str, status: DocumentStatus, details: str = "") -> bool:
        fields: dict[str, Any] = {"status": status.value, "updated_at": utc_now()}
        if details:
            fields["stage_details"] = details
        if status is DocumentStatus.COMPLETED:
            fields.update({"progress_percentage": 100, "current_stage": "completed"})
        elif status is DocumentStatus.FAILED:
            fields["current_stage"] = "failed"
        result = self.collection.update_one(_document_filter(document_id), {"$set": fields})
        return result.matched_count > 0

    def update_progress(self, document_id: str, percentage: int, stage: str, details: str = "") -> bool:
        if not 0 <= percentage <= 100:
            raise ValueError("Document progress must be between 0 and 100.")
        result = self.collection.update_one(
            _document_filter(document_id),
            {
                "$set": {
                    "progress_percentage": percentage,
                    "current_stage": stage,
                    "stage_details": details,
                    "updated_at": utc_now(),
                }
            },
        )
        return result.matched_count > 0

    def begin_indexing(self, document_id: str, details: str = "") -> bool:
        """Mark a document unavailable while all indexes are being replaced."""

        document_filter: dict[str, Any] = _document_filter(document_id)
        document_filter["status"] = {"$ne": DocumentStatus.PROCESSING.value}
        result = self.collection.update_one(
            document_filter,
            {
                "$set": {
                    "status": DocumentStatus.PROCESSING.value,
                    "progress_percentage": 0,
                    "current_stage": "queued",
                    "stage_details": details,
                    "updated_at": utc_now(),
                }
            },
        )
        return result.matched_count > 0

    def complete_indexing(
        self,
        document_id: str,
        *,
        index_version: str,
        chunking_version: str,
        embedding_model: str,
        chunk_count: int,
    ) -> bool:
        """Complete ingestion and record the exact index configuration."""

        now = utc_now()
        result = self.collection.update_one(
            _document_filter(document_id),
            {
                "$set": {
                    "status": DocumentStatus.COMPLETED.value,
                    "progress_percentage": 100,
                    "current_stage": "completed",
                    "stage_details": f"{chunk_count} chunks indexed",
                    "index_version": index_version,
                    "chunking_version": chunking_version,
                    "embedding_model": embedding_model,
                    "indexed_at": now,
                    "chunk_count": chunk_count,
                    "updated_at": now,
                }
            },
        )
        return result.matched_count > 0


class ChunkRepository:
    """Persist structured chunks produced by document ingestion."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        self.collection = collection

    @classmethod
    def from_store(cls, store: MongoStore) -> "ChunkRepository":
        return cls(store.collection("chunks"))

    def replace_document_chunks(self, document_id: str, chunks: list[Chunk]) -> list[ChunkRecord]:
        self.collection.delete_many({"document_id": document_id})
        created_at = utc_now()
        payloads = [
            {
                "_id": chunk.chunk_id,
                "document_id": document_id,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type.value,
                "index": chunk.index,
                "metadata": chunk.metadata,
                "created_at": created_at,
            }
            for chunk in chunks
        ]
        if payloads:
            self.collection.insert_many(payloads)
        return [_chunk_record(payload) for payload in payloads]

    def list_for_document(self, document_id: str) -> list[ChunkRecord]:
        cursor = self.collection.find({"document_id": document_id}).sort("index", 1)
        return [_chunk_record(chunk) for chunk in cursor]


class ConversationRepository:
    """Persist conversations and their ordered messages."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        self.collection = collection

    @classmethod
    def from_store(cls, store: MongoStore) -> "ConversationRepository":
        return cls(store.collection("conversations"))

    def create(
        self,
        title: str = "New conversation",
        assistant_id: str | None = None,
        document_id: str | None = None,
    ) -> ConversationRecord:
        now = utc_now()
        payload: dict[str, Any] = {
            "_id": str(uuid4()),
            "title": title.strip() or "New conversation",
            "assistant_id": assistant_id,
            "document_id": document_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        self.collection.insert_one(payload)
        return _conversation_record(payload)

    def get(self, conversation_id: str) -> ConversationRecord | None:
        conversation = self.collection.find_one({"_id": conversation_id})
        return _conversation_record(conversation) if conversation else None

    def list(self, *, skip: int = 0, limit: int = 100) -> list[ConversationRecord]:
        if skip < 0 or not 1 <= limit <= 100:
            raise ValueError("Conversation pagination is out of range.")
        cursor = self.collection.find({}).sort("updated_at", -1).skip(skip).limit(limit)
        return [_conversation_record(conversation) for conversation in cursor]

    def add_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation message role must be 'user' or 'assistant'.")
        if not content.strip():
            raise ValueError("Conversation message content must not be empty.")
        message = {
            "message_id": str(uuid4()),
            "role": role,
            "content": content.strip(),
            "timestamp": utc_now(),
            "metadata": metadata or {},
        }
        result = self.collection.update_one(
            {"_id": conversation_id},
            {"$push": {"messages": message}, "$set": {"updated_at": message["timestamp"]}},
        )
        if result.matched_count == 0:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return _conversation_message(message)

    def add_turn(
        self,
        conversation_id: str,
        *,
        user_content: str,
        assistant_content: str,
        assistant_metadata: dict[str, Any] | None = None,
    ) -> tuple[ConversationMessage, ConversationMessage]:
        if not user_content.strip() or not assistant_content.strip():
            raise ValueError("Conversation turn messages must not be empty.")
        now = utc_now()
        messages = [
            {
                "message_id": str(uuid4()),
                "role": "user",
                "content": user_content.strip(),
                "timestamp": now,
                "metadata": {},
            },
            {
                "message_id": str(uuid4()),
                "role": "assistant",
                "content": assistant_content.strip(),
                "timestamp": now,
                "metadata": assistant_metadata or {},
            },
        ]
        result = self.collection.update_one(
            {"_id": conversation_id},
            {"$push": {"messages": {"$each": messages}}, "$set": {"updated_at": now}},
        )
        if result.matched_count == 0:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return _conversation_message(messages[0]), _conversation_message(messages[1])


def _document_filter(document_id: str) -> dict[str, ObjectId]:
    try:
        return {"_id": ObjectId(document_id)}
    except InvalidId as exc:
        raise ValueError(f"Invalid document ID: {document_id}") from exc


def _document_record(document: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        document_id=str(document["_id"]),
        title=str(document["title"]),
        file_type=str(document["file_type"]),
        file_path=str(document["file_path"]),
        file_size=int(document["file_size"]),
        file_hash=str(document["file_hash"]),
        status=DocumentStatus(str(document["status"])),
        progress_percentage=int(document.get("progress_percentage", 0)),
        current_stage=str(document.get("current_stage", "")),
        stage_details=str(document.get("stage_details", "")),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        metadata=dict(document.get("metadata") or {}),
        index_version=document.get("index_version"),
        chunking_version=document.get("chunking_version"),
        embedding_model=document.get("embedding_model"),
        indexed_at=document.get("indexed_at"),
        chunk_count=int(document.get("chunk_count", 0)),
    )


def _conversation_message(message: dict[str, Any]) -> ConversationMessage:
    return ConversationMessage(
        message_id=str(message["message_id"]),
        role=str(message["role"]),
        content=str(message["content"]),
        timestamp=message["timestamp"],
        metadata=dict(message.get("metadata") or {}),
    )


def _chunk_record(chunk: dict[str, Any]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(chunk["_id"]),
        document_id=str(chunk["document_id"]),
        text=str(chunk["text"]),
        chunk_type=str(chunk["chunk_type"]),
        index=int(chunk["index"]),
        metadata=dict(chunk.get("metadata") or {}),
        created_at=chunk["created_at"],
    )


def _conversation_record(conversation: dict[str, Any]) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=str(conversation["_id"]),
        title=str(conversation.get("title") or "New conversation"),
        assistant_id=conversation.get("assistant_id"),
        document_id=conversation.get("document_id"),
        messages=[_conversation_message(message) for message in conversation.get("messages", [])],
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
    )
