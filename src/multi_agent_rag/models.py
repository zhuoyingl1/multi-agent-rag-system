"""Core data models for documents and structured chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256


class RetrievalType(str, Enum):
    """Retrieval signal types used by the local hybrid retriever."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    ENTITY = "entity"


class ChunkType(str, Enum):
    """Supported chunk categories used by the local retrieval pipeline."""

    PROSE = "prose"
    CODE = "code"
    FORMULA = "formula"
    TABLE = "table"


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    title: str
    text: str
    document_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def stable_id(self) -> str:
        if self.document_id:
            return self.document_id
        digest = sha256(f"{self.title}\n{self.text}".encode("utf-8")).hexdigest()[:16]
        return f"doc_{digest}"


@dataclass(frozen=True)
class Chunk:
    """A structured document chunk with stable identity and source metadata."""

    document_id: str
    chunk_id: str
    text: str
    chunk_type: ChunkType
    index: int
    metadata: dict[str, str] = field(default_factory=dict)


def stable_chunk_id(document_id: str, index: int, chunk_type: ChunkType, text: str) -> str:
    digest = sha256(f"{document_id}\n{index}\n{chunk_type.value}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


@dataclass(frozen=True)
class SearchResult:
    """A scored retrieval result with lightweight highlights."""

    chunk: Chunk
    score: float
    retrieval_type: RetrievalType
    highlights: list[str] = field(default_factory=list)
