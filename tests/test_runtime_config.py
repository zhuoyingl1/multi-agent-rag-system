from datetime import UTC, datetime

import pytest

from multi_agent_rag.runtime_config import RuntimeSettings, RuntimeSettingsSnapshot, RuntimeSettingsStore


class FakePersistence:
    def __init__(self, snapshot: RuntimeSettingsSnapshot | None = None) -> None:
        self.current = snapshot
        self.load_count = 0
        self.saved_revisions: list[int] = []

    def load(self) -> RuntimeSettingsSnapshot | None:
        self.load_count += 1
        return self.current

    def save(self, settings: RuntimeSettings, expected_revision: int) -> RuntimeSettingsSnapshot:
        self.saved_revisions.append(expected_revision)
        self.current = RuntimeSettingsSnapshot(settings, expected_revision + 1, datetime.now(UTC))
        return self.current


def test_runtime_settings_load_supported_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "false")
    monkeypatch.setenv("RAG_CONTEXT_BUDGET_TOKENS", "2400")

    settings = RuntimeSettings.from_env()

    assert settings.top_k == 7
    assert settings.query_rewrite_enabled is False
    assert settings.context_budget_tokens == 2400


def test_runtime_settings_store_applies_partial_updates() -> None:
    store = RuntimeSettingsStore(RuntimeSettings())

    updated = store.update({"top_k": 8, "query_rewrite_enabled": False})

    assert updated.revision == 1
    assert updated.source == "runtime"
    assert updated.updated_at is not None
    assert updated.settings.top_k == 8
    assert updated.settings.query_rewrite_enabled is False
    assert updated.settings.context_budget_tokens == 1600


def test_runtime_settings_store_rejects_invalid_cross_field_update() -> None:
    store = RuntimeSettingsStore(RuntimeSettings())

    with pytest.raises(ValueError, match="dynamic_top_k_min"):
        store.update({"dynamic_top_k_min": 9, "dynamic_top_k_max": 4})

    snapshot = store.snapshot()
    assert snapshot.revision == 0
    assert snapshot.settings.dynamic_top_k_min == 3
    assert snapshot.settings.dynamic_top_k_max == 8


def test_runtime_settings_reject_invalid_boolean_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="QUERY_REWRITE_ENABLED"):
        RuntimeSettings.from_env()


def test_runtime_settings_store_refreshes_shared_settings_after_ttl() -> None:
    now = [100.0]
    persistence = FakePersistence(RuntimeSettingsSnapshot(RuntimeSettings(top_k=7), 1, datetime.now(UTC)))
    store = RuntimeSettingsStore(
        RuntimeSettings(),
        persistence=persistence,
        cache_ttl_seconds=5,
        clock=lambda: now[0],
    )

    assert store.snapshot().settings.top_k == 7
    persistence.current = RuntimeSettingsSnapshot(RuntimeSettings(top_k=9), 2, datetime.now(UTC))
    now[0] += 4
    assert store.snapshot().settings.top_k == 7
    now[0] += 1
    assert store.snapshot().settings.top_k == 9
    assert persistence.load_count == 2
    assert store.backend == "mongodb"
    assert store.scope == "shared"


def test_runtime_settings_store_refreshes_before_persistent_update() -> None:
    persistence = FakePersistence(RuntimeSettingsSnapshot(RuntimeSettings(top_k=7), 4, datetime.now(UTC)))
    store = RuntimeSettingsStore(RuntimeSettings(), persistence=persistence, cache_ttl_seconds=60)

    updated = store.update({"context_budget_tokens": 2400})

    assert persistence.saved_revisions == [4]
    assert updated.revision == 5
    assert updated.settings.top_k == 7
    assert updated.settings.context_budget_tokens == 2400
