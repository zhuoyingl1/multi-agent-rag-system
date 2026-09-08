"""Persistent document registration and background ingestion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from multi_agent_rag.documents import load_document
from multi_agent_rag.persistence import ChunkRepository, DocumentRecord, DocumentRepository, DocumentStatus
from multi_agent_rag.retrieval.chunking import chunk_document


@dataclass(frozen=True)
class RegisteredDocument:
    """A document registration result, including duplicate detection."""

    document: DocumentRecord
    duplicate: bool


class DocumentIngestionService:
    """Coordinate document metadata, parsing, chunking, and status updates."""

    def __init__(self, documents: DocumentRepository, chunks: ChunkRepository) -> None:
        self.documents = documents
        self.chunks = chunks

    def register(
        self,
        *,
        title: str,
        file_type: str,
        file_path: str,
        file_size: int,
        file_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredDocument:
        duplicate = self.documents.find_duplicate(file_hash)
        if duplicate is not None:
            return RegisteredDocument(document=duplicate, duplicate=True)
        document = self.documents.create(
            title=title,
            file_type=file_type,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            metadata=metadata,
        )
        return RegisteredDocument(document=document, duplicate=False)

    def process(self, document_id: str, file_path: str | Path) -> int:
        try:
            self.documents.update_progress(document_id, 20, "parsing")
            document = replace(load_document(file_path), document_id=document_id)
            self.documents.update_progress(document_id, 60, "chunking")
            chunks = chunk_document(document)
            self.documents.update_progress(document_id, 85, "storing", f"{len(chunks)} chunks")
            self.chunks.replace_document_chunks(document_id, chunks)
            self.documents.update_status(document_id, DocumentStatus.COMPLETED)
            return len(chunks)
        except Exception as exc:
            self.documents.update_status(document_id, DocumentStatus.FAILED, str(exc))
            return 0

    def get(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)
