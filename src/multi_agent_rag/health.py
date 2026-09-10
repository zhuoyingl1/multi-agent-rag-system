"""Runtime dependency probes for deployment health checks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from time import perf_counter
from urllib import request as urlrequest

from multi_agent_rag import __version__

DEFAULT_REQUIRED_SERVICES = ("mongodb", "qdrant", "neo4j", "ollama", "task_queue")
Probe = Callable[[float], str]


@dataclass(frozen=True)
class DependencyStatus:
    """Result of one live dependency probe."""

    name: str
    status: str
    latency_ms: float
    details: str


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregated readiness state for required runtime dependencies."""

    status: str
    version: str
    checked_at: str
    duration_ms: float
    required_services: list[str]
    dependencies: list[DependencyStatus]

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_readiness(
    required_services: list[str] | None = None,
    *,
    timeout_seconds: float | None = None,
    probes: Mapping[str, Probe] | None = None,
) -> ReadinessReport:
    """Probe required services concurrently and return deployment readiness."""

    services = required_services or _required_services_from_env()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.getenv("HEALTH_PROBE_TIMEOUT_SECONDS") or "2")
    )
    if timeout <= 0:
        raise ValueError("Health probe timeout must be greater than zero.")
    available_probes = dict(_default_probes() if probes is None else probes)
    started_at = perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(services))) as executor:
        futures = {
            name: executor.submit(_run_probe, name, available_probes.get(name), timeout)
            for name in services
        }
        statuses = [futures[name].result() for name in services]
    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    ready = all(item.status == "ready" for item in statuses)
    return ReadinessReport(
        status="ready" if ready else "not_ready",
        version=__version__,
        checked_at=datetime.now(UTC).isoformat(),
        duration_ms=duration_ms,
        required_services=services,
        dependencies=statuses,
    )


def _required_services_from_env() -> list[str]:
    configured = os.getenv("HEALTH_REQUIRED_SERVICES") or ",".join(DEFAULT_REQUIRED_SERVICES)
    return list(dict.fromkeys(name.strip().lower() for name in configured.split(",") if name.strip()))


def _run_probe(name: str, probe: Probe | None, timeout: float) -> DependencyStatus:
    started_at = perf_counter()
    if probe is None:
        return DependencyStatus(name, "unavailable", 0.0, "No health probe is registered for this service.")
    try:
        details = probe(timeout)
        status = "ready"
    except Exception as exc:
        details = f"Probe failed with {exc.__class__.__name__}."
        status = "unavailable"
    return DependencyStatus(name, status, round((perf_counter() - started_at) * 1000, 2), details)


def _default_probes() -> dict[str, Probe]:
    return {
        "mongodb": _probe_mongodb,
        "qdrant": _probe_qdrant,
        "neo4j": _probe_neo4j,
        "ollama": _probe_ollama,
        "task_queue": _probe_task_queue,
    }


def _probe_mongodb(timeout: float) -> str:
    from pymongo import MongoClient

    client = MongoClient(
        os.getenv("MONGODB_URI") or "mongodb://localhost:27017",
        serverSelectionTimeoutMS=max(100, int(timeout * 1000)),
    )
    try:
        client.admin.command("ping")
    finally:
        client.close()
    return "MongoDB ping succeeded."


def _probe_qdrant(timeout: float) -> str:
    base_url = (os.getenv("QDRANT_URL") or "http://localhost:6333").rstrip("/")
    with urlrequest.urlopen(f"{base_url}/healthz", timeout=timeout) as response:
        if not 200 <= int(response.status) < 300:
            raise RuntimeError("Qdrant returned a non-success status.")
    return "Qdrant health check succeeded."


def _probe_neo4j(timeout: float) -> str:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI") or "bolt://localhost:7687",
        auth=(os.getenv("NEO4J_USER") or "neo4j", os.getenv("NEO4J_PASSWORD") or "password123"),
        connection_timeout=timeout,
    )
    try:
        driver.verify_connectivity()
    finally:
        driver.close()
    return "Neo4j connectivity check succeeded."


def _probe_ollama(timeout: float) -> str:
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    model_name = os.getenv("LLM_ANSWER_MODEL") or "qwen2.5:3b"
    with urlrequest.urlopen(f"{base_url}/api/tags", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = {str(model.get("name") or "") for model in payload.get("models", []) if isinstance(model, dict)}
    if model_name not in models:
        raise RuntimeError("Configured Ollama model is not available.")
    return f"Ollama model {model_name} is available."


def _probe_task_queue(timeout: float) -> str:
    from multi_agent_rag.task_queue import check_task_queue

    status = check_task_queue(timeout)
    if not status.ready:
        raise RuntimeError(status.details)
    return status.details
