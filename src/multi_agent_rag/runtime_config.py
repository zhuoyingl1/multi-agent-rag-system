"""Validated process-scoped runtime settings for retrieval requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import os
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    """Non-sensitive retrieval settings that may change between requests."""

    top_k: int = 5
    query_rewrite_enabled: bool = True
    query_rewrite_max_variants: int = 3
    dynamic_top_k_enabled: bool = True
    dynamic_top_k_min: int = 3
    dynamic_top_k_max: int = 8
    dynamic_top_k_gap_high: float = 2.0
    dynamic_top_k_gap_low: float = 0.6
    context_budget_tokens: int = 1600
    reranker_candidate_multiplier: int = 3
    rrf_k: float = 60.0

    def __post_init__(self) -> None:
        _validate_range("top_k", self.top_k, 1, 20)
        _validate_range("query_rewrite_max_variants", self.query_rewrite_max_variants, 1, 5)
        _validate_range("dynamic_top_k_min", self.dynamic_top_k_min, 1, 20)
        _validate_range("dynamic_top_k_max", self.dynamic_top_k_max, 1, 20)
        _validate_range("dynamic_top_k_gap_high", self.dynamic_top_k_gap_high, 0.0, 10.0)
        _validate_range("dynamic_top_k_gap_low", self.dynamic_top_k_gap_low, 0.0, 10.0)
        _validate_range("context_budget_tokens", self.context_budget_tokens, 128, 32000)
        _validate_range("reranker_candidate_multiplier", self.reranker_candidate_multiplier, 1, 10)
        _validate_range("rrf_k", self.rrf_k, 1.0, 200.0)
        if self.dynamic_top_k_min > self.dynamic_top_k_max:
            raise ValueError("dynamic_top_k_min must not exceed dynamic_top_k_max.")
        if self.dynamic_top_k_gap_low > self.dynamic_top_k_gap_high:
            raise ValueError("dynamic_top_k_gap_low must not exceed dynamic_top_k_gap_high.")

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        """Load the initial process settings from environment variables."""

        return cls(
            top_k=int(os.getenv("RAG_TOP_K") or "5"),
            query_rewrite_enabled=_env_bool("QUERY_REWRITE_ENABLED", True),
            query_rewrite_max_variants=int(os.getenv("QUERY_REWRITE_MAX_VARIANTS") or "3"),
            dynamic_top_k_enabled=_env_bool("DYNAMIC_TOP_K_ENABLED", True),
            dynamic_top_k_min=int(os.getenv("DYNAMIC_TOP_K_MIN") or "3"),
            dynamic_top_k_max=int(os.getenv("DYNAMIC_TOP_K_MAX") or "8"),
            dynamic_top_k_gap_high=float(os.getenv("DYNAMIC_TOP_K_GAP_HIGH") or "2.0"),
            dynamic_top_k_gap_low=float(os.getenv("DYNAMIC_TOP_K_GAP_LOW") or "0.6"),
            context_budget_tokens=int(os.getenv("RAG_CONTEXT_BUDGET_TOKENS") or "1600"),
            reranker_candidate_multiplier=int(os.getenv("RERANKER_CANDIDATE_MULTIPLIER") or "3"),
            rrf_k=float(os.getenv("RRF_K") or "60"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSettingsSnapshot:
    """One immutable version of the active process settings."""

    settings: RuntimeSettings
    revision: int
    updated_at: datetime | None

    @property
    def source(self) -> str:
        return "runtime" if self.revision else "environment"


class RuntimeSettingsStore:
    """Apply atomic runtime updates and provide request-stable snapshots."""

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self._lock = RLock()
        self._settings = settings or RuntimeSettings.from_env()
        self._revision = 0
        self._updated_at: datetime | None = None

    def snapshot(self) -> RuntimeSettingsSnapshot:
        with self._lock:
            return RuntimeSettingsSnapshot(self._settings, self._revision, self._updated_at)

    def update(self, changes: dict[str, Any]) -> RuntimeSettingsSnapshot:
        if not changes:
            return self.snapshot()
        with self._lock:
            updated = replace(self._settings, **changes)
            self._settings = updated
            self._revision += 1
            self._updated_at = datetime.now(UTC)
            return RuntimeSettingsSnapshot(updated, self._revision, self._updated_at)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _validate_range(name: str, value: float, minimum: float, maximum: float) -> None:
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
