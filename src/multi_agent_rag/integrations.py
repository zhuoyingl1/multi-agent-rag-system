"""Production integration readiness helpers."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IntegrationConfig:
    """Configuration values for optional production integrations."""

    qdrant_url: str | None = None
    qdrant_collection: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_database: str | None = None
    reranker_model: str | None = None
    enable_langgraph: bool = False

    @classmethod
    def from_env(cls) -> "IntegrationConfig":
        return cls(
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_database=os.getenv("NEO4J_DATABASE"),
            reranker_model=os.getenv("RERANKER_MODEL"),
            enable_langgraph=os.getenv("ENABLE_LANGGRAPH", "").lower() in {"1", "true", "yes"},
        )


@dataclass(frozen=True)
class IntegrationStatus:
    """Readiness status for one integration boundary."""

    name: str
    role: str
    status: str
    required_package: str | None
    configured: bool
    package_available: bool
    notes: str


@dataclass(frozen=True)
class IntegrationReport:
    """Readiness report for local and optional production paths."""

    mode: str
    ready_count: int
    integration_count: int
    integrations: list[IntegrationStatus]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def check_integrations(config: IntegrationConfig | None = None) -> IntegrationReport:
    current = config or IntegrationConfig.from_env()
    statuses = [
        IntegrationStatus(
            name="local_hybrid_store",
            role="Default deterministic keyword, vector-like, and entity retrieval",
            status="ready",
            required_package=None,
            configured=True,
            package_available=True,
            notes="Used by default for local development, tests, and demos.",
        ),
        _status(
            name="qdrant",
            role="Production vector search backend",
            required_package="qdrant_client",
            configured=bool(current.qdrant_url and current.qdrant_collection),
            config_note="Set QDRANT_URL and QDRANT_COLLECTION to enable this path.",
        ),
        _status(
            name="neo4j",
            role="Entity graph expansion and relationship-aware retrieval",
            required_package="neo4j",
            configured=bool(current.neo4j_uri and current.neo4j_user),
            config_note="Set NEO4J_URI and NEO4J_USER to enable this path.",
        ),
        _status(
            name="bge_reranker",
            role="Cross-encoder or BGE-style reranking for retrieved candidates",
            required_package="sentence_transformers",
            configured=bool(current.reranker_model),
            config_note="Set RERANKER_MODEL to enable this path.",
        ),
        _status(
            name="langgraph",
            role="Production multi-agent orchestration graph",
            required_package="langgraph",
            configured=current.enable_langgraph,
            config_note="Set ENABLE_LANGGRAPH=true to enable this path.",
        ),
    ]
    ready_count = len([status for status in statuses if status.status == "ready"])
    mode = "production_ready" if ready_count == len(statuses) else "local_with_optional_integrations"
    return IntegrationReport(
        mode=mode,
        ready_count=ready_count,
        integration_count=len(statuses),
        integrations=statuses,
    )


def _status(name: str, role: str, required_package: str, configured: bool, config_note: str) -> IntegrationStatus:
    package_available = importlib.util.find_spec(required_package) is not None
    if configured and package_available:
        status = "ready"
        notes = "Configuration and optional package are available."
    elif configured:
        status = "missing_package"
        notes = f"Install the production extra or package '{required_package}'."
    else:
        status = "missing_config"
        notes = config_note
    return IntegrationStatus(
        name=name,
        role=role,
        status=status,
        required_package=required_package,
        configured=configured,
        package_available=package_available,
        notes=notes,
    )
