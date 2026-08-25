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
    RERANKED = "reranked"


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


@dataclass(frozen=True)
class AgentPlan:
    """Planner output describing selected agents and their tasks."""

    selected_agents: list[str]
    tasks: dict[str, str]
    reasoning: str


@dataclass(frozen=True)
class Coordination:
    """Coordinator output that binds a plan to retrieved evidence."""

    selected_agents: list[str]
    tasks: dict[str, str]
    sources: list[SearchResult]
    evidence_count: int
    reasoning: str


@dataclass(frozen=True)
class AgentResult:
    """Specialist agent output grounded in retrieved sources."""

    agent_name: str
    task: str
    content: str
    confidence: float
    sources: list[SearchResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class JudgeResult:
    """Grounding judge result for a draft answer."""

    score: float
    reason: str
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowResult:
    """End-to-end workflow response."""

    query: str
    answer: str
    plan: AgentPlan
    agents: list[AgentResult]
    grounding: JudgeResult
    sources: list[SearchResult]
    metrics: dict[str, float | int | str]
