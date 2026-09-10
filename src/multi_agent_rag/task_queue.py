"""Configurable dispatch for document ingestion tasks."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

LOCAL_BACKENDS = {"local", "background", "in_process"}


class DocumentTaskDispatchError(RuntimeError):
    """Raised when a document task cannot be queued."""


@dataclass(frozen=True)
class TaskDispatch:
    """Result of submitting a document ingestion task."""

    backend: str
    task_id: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class TaskQueueStatus:
    """Live status of the configured document task backend."""

    backend: str
    status: str
    details: str
    worker_count: int | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def configured_task_backend() -> str:
    return (os.getenv("DOCUMENT_TASK_BACKEND") or "local").strip().lower()


def dispatch_document_task(
    background_tasks: Any,
    document_id: str,
    file_path: str,
    local_processor: Callable[[str, str], object],
    *,
    celery_task: Any | None = None,
) -> TaskDispatch:
    """Submit ingestion locally or through Celery without changing API callers."""

    backend = configured_task_backend()
    if backend in LOCAL_BACKENDS:
        background_tasks.add_task(local_processor, document_id, file_path)
        return TaskDispatch(backend="local")
    if backend != "celery":
        raise ValueError(f"Unsupported document task backend: {backend}")

    try:
        task = celery_task or _document_task()
        queued = task.delay(document_id, file_path)
        return TaskDispatch(backend="celery", task_id=str(queued.id))
    except Exception as exc:
        reason = f"Celery dispatch failed with {exc.__class__.__name__}."
        if _env_flag("DOCUMENT_TASK_FALLBACK_LOCAL", False):
            background_tasks.add_task(local_processor, document_id, file_path)
            return TaskDispatch(backend="local", fallback_reason=reason)
        raise DocumentTaskDispatchError(reason) from exc


def check_task_queue(timeout_seconds: float = 2.0) -> TaskQueueStatus:
    """Verify the configured queue, Redis broker, and at least one Celery worker."""

    backend = configured_task_backend()
    if backend in LOCAL_BACKENDS:
        return TaskQueueStatus("local", "ready", "In-process document tasks are enabled.")
    if backend != "celery":
        return TaskQueueStatus(backend, "unavailable", "The configured task backend is not supported.")

    redis_client = None
    try:
        from redis import Redis

        from multi_agent_rag.celery_app import celery_app

        redis_client = Redis.from_url(
            os.getenv("CELERY_BROKER_URL") or "redis://localhost:6379/0",
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        redis_client.ping()
        replies = celery_app.control.inspect(timeout=timeout_seconds).ping() or {}
        if not replies:
            return TaskQueueStatus("celery", "unavailable", "Redis is ready, but no Celery worker responded.", 0)
        return TaskQueueStatus("celery", "ready", "Redis and Celery workers are ready.", len(replies))
    except Exception as exc:
        return TaskQueueStatus(
            "celery",
            "unavailable",
            f"Task queue probe failed with {exc.__class__.__name__}.",
            0,
        )
    finally:
        if redis_client is not None:
            redis_client.close()


def _document_task() -> Any:
    from multi_agent_rag.document_tasks import process_document

    return process_document


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
