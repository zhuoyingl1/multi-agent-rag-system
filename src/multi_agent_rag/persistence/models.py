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
