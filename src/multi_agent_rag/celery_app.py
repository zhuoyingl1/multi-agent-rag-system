"""Celery application for durable background document processing."""

from __future__ import annotations

import os

from celery import Celery


def create_celery_app() -> Celery:
    broker_url = os.getenv("CELERY_BROKER_URL") or "redis://localhost:6379/0"
    result_backend = os.getenv("CELERY_RESULT_BACKEND") or "redis://localhost:6379/1"
    application = Celery(
        "multi_agent_rag",
        broker=broker_url,
        backend=result_backend,
        include=["multi_agent_rag.document_tasks"],
    )
    application.conf.update(
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        result_serializer="json",
        task_acks_late=True,
        task_serializer="json",
        task_track_started=True,
        worker_prefetch_multiplier=1,
    )
    return application


celery_app = create_celery_app()
