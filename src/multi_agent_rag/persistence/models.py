"""Persistent document and conversation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DocumentStatus(str, Enum):
    """Lifecycle states for an uploaded document."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class UserRecord:
    """A registered API user with a one-way password hash."""

    user_id: str
    email: str
    display_name: str
    password_hash: str
    active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class KnowledgeSpaceRecord:
    """A named collection of documents used as one retrieval scope."""

    knowledge_space_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    owner_id: str | None = None


@dataclass(frozen=True)
class DocumentRecord:
    """Document metadata stored before and during ingestion."""

    document_id: str
    title: str
    file_type: str
    file_path: str
    file_size: int
    file_hash: str
    status: DocumentStatus
    progress_percentage: int
    current_stage: str
    stage_details: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    index_version: str | None = None
    chunking_version: str | None = None
    embedding_model: str | None = None
    indexed_at: datetime | None = None
    chunk_count: int = 0
    knowledge_space_id: str | None = None
    owner_id: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    """One structured document chunk stored for later indexing."""

    chunk_id: str
    document_id: str
    text: str
    chunk_type: str
    index: int
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ConversationMessage:
    """One persisted user or assistant message."""

    message_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationRecord:
    """Conversation metadata and its ordered messages."""

    conversation_id: str
    title: str
    messages: list[ConversationMessage]
    created_at: datetime
    updated_at: datetime
    assistant_id: str | None = None
    document_id: str | None = None
    knowledge_space_id: str | None = None
    owner_id: str | None = None
