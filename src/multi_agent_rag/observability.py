"""Lightweight runtime metrics for local API and workflow demos."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    """In-memory metrics registry for prototype observability."""

    run_count: int = 0
    total_latency_ms: float = 0.0
    total_retrieved_sources: int = 0
    total_failed_agents: int = 0
    grounding_scores: list[float] = field(default_factory=list)

    def record_run(self, metrics: dict[str, float | int | str]) -> None:
        self.run_count += 1
        self.total_latency_ms += float(metrics.get("latency_ms", 0.0))
        self.total_retrieved_sources += int(metrics.get("retrieved_sources", 0))
        self.total_failed_agents += int(metrics.get("failed_agents", 0))
        self.grounding_scores.append(float(metrics.get("grounding_score", 0.0)))

    def snapshot(self) -> dict[str, float | int]:
        average_latency = self.total_latency_ms / self.run_count if self.run_count else 0.0
        average_grounding = sum(self.grounding_scores) / len(self.grounding_scores) if self.grounding_scores else 0.0
        average_sources = self.total_retrieved_sources / self.run_count if self.run_count else 0.0
        return {
            "run_count": self.run_count,
            "average_latency_ms": round(average_latency, 2),
            "average_grounding_score": round(average_grounding, 3),
            "average_retrieved_sources": round(average_sources, 2),
            "total_failed_agents": self.total_failed_agents,
        }


metrics_registry = MetricsRegistry()
