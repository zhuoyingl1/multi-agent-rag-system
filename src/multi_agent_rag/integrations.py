"""Production integration readiness helpers."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IntegrationConfig:
    """Configuration values for optional production integrations."""

    qdrant_url: str | None = "http://localhost:6333"
    qdrant_collection: str | None = "documents"
    neo4j_uri: str | None = "bolt://localhost:7687"
    neo4j_user: str | None = "neo4j"
    neo4j_database: str | None = "neo4j"
    reranker_model: str | None = None

    @classmethod
    def from_env(cls) -> "IntegrationConfig":
        return cls(
            qdrant_url=os.getenv("QDRANT_URL") or "http://localhost:6333",
            qdrant_collection=os.getenv("QDRANT_COLLECTION") or "documents",
            neo4j_uri=os.getenv("NEO4J_URI") or "bolt://localhost:7687",
            neo4j_user=os.getenv("NEO4J_USER") or "neo4j",
            neo4j_database=os.getenv("NEO4J_DATABASE") or "neo4j",
            reranker_model=os.getenv("RERANKER_MODEL"),
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
        _reranker_status(current.reranker_model),
        _status(
            name="langgraph",
            role="Production multi-agent orchestration graph",
            required_package="langgraph",
            configured=True,
            config_note="Install the base project dependencies to enable this path.",
        ),
    ]
    ready_count = len([status for status in statuses if status.status == "ready"])
    reranker_status = next(status for status in statuses if status.name == "bge_reranker")
    production_reranker_ready = reranker_status.status == "ready" and reranker_status.required_package == "sentence_transformers"
    mode = "production_ready" if ready_count == len(statuses) and production_reranker_ready else "local_with_optional_integrations"
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
        notes = "Configuration and package are available."
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


def _reranker_status(model_name: str | None) -> IntegrationStatus:
    role = "Cross-encoder or BGE-style reranking for retrieved candidates"
    if not model_name:
        return IntegrationStatus(
            name="bge_reranker",
            role=role,
            status="missing_config",
            required_package="sentence_transformers",
            configured=False,
            package_available=importlib.util.find_spec("sentence_transformers") is not None,
            notes="Set RERANKER_MODEL to enable this path.",
        )
    if model_name.lower() in {"local", "lexical", "deterministic"}:
        return IntegrationStatus(
            name="bge_reranker",
            role=role,
            status="ready",
            required_package=None,
            configured=True,
            package_available=True,
            notes="Built-in deterministic reranking is configured for local inspection.",
        )
    return _status(
        name="bge_reranker",
        role=role,
        required_package="sentence_transformers",
        configured=True,
        config_note="Set RERANKER_MODEL to enable this path.",
    )
