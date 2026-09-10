from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from multi_agent_rag.task_queue import (
    DocumentTaskDispatchError,
    check_task_queue,
    dispatch_document_task,
)


class StubBackgroundTasks:
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[object, ...]]] = []

    def add_task(self, function, *args) -> None:
        self.calls.append((function, args))


def test_local_dispatch_uses_fastapi_background_task(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_TASK_BACKEND", "local")
    background_tasks = StubBackgroundTasks()
    processor = MagicMock()

    result = dispatch_document_task(background_tasks, "document-1", "notes.md", processor)

    assert result.backend == "local"
    assert result.task_id is None
    assert background_tasks.calls == [(processor, ("document-1", "notes.md"))]


def test_celery_dispatch_returns_task_identifier(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_TASK_BACKEND", "celery")
    background_tasks = StubBackgroundTasks()
    task = MagicMock()
    task.delay.return_value = SimpleNamespace(id="task-123")

    result = dispatch_document_task(background_tasks, "document-1", "notes.md", MagicMock(), celery_task=task)

    assert result.backend == "celery"
    assert result.task_id == "task-123"
    assert background_tasks.calls == []
    task.delay.assert_called_once_with("document-1", "notes.md")


def test_celery_dispatch_fails_closed_by_default(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_TASK_BACKEND", "celery")
    monkeypatch.delenv("DOCUMENT_TASK_FALLBACK_LOCAL", raising=False)
    task = MagicMock()
    task.delay.side_effect = ConnectionError("redis://user:secret@private-host")

    with pytest.raises(DocumentTaskDispatchError, match="ConnectionError") as exc_info:
        dispatch_document_task(StubBackgroundTasks(), "document-1", "notes.md", MagicMock(), celery_task=task)

    assert "secret" not in str(exc_info.value)


def test_celery_dispatch_can_fall_back_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_TASK_BACKEND", "celery")
    monkeypatch.setenv("DOCUMENT_TASK_FALLBACK_LOCAL", "true")
    background_tasks = StubBackgroundTasks()
    processor = MagicMock()
    task = MagicMock()
    task.delay.side_effect = ConnectionError("offline")

    result = dispatch_document_task(background_tasks, "document-1", "notes.md", processor, celery_task=task)

    assert result.backend == "local"
    assert result.fallback_reason == "Celery dispatch failed with ConnectionError."
    assert background_tasks.calls == [(processor, ("document-1", "notes.md"))]


def test_local_queue_health_is_ready(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_TASK_BACKEND", "local")

    status = check_task_queue()

    assert status.ready is True
    assert status.backend == "local"
    assert status.worker_count is None


def test_unknown_queue_backend_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_TASK_BACKEND", "unknown")

    status = check_task_queue()

    assert status.ready is False
    assert status.backend == "unknown"
