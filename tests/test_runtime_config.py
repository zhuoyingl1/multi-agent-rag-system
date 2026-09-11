import pytest

from multi_agent_rag.runtime_config import RuntimeSettings, RuntimeSettingsStore


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
