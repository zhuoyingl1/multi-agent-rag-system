from multi_agent_rag.observability import MetricsRegistry


def test_metrics_registry_records_runs() -> None:
    registry = MetricsRegistry()

    registry.record_run({"latency_ms": 10.0, "retrieved_sources": 3, "failed_agents": 1, "grounding_score": 0.5})
    registry.record_run({"latency_ms": 20.0, "retrieved_sources": 5, "failed_agents": 0, "grounding_score": 1.0})

    snapshot = registry.snapshot()
    assert snapshot["run_count"] == 2
    assert snapshot["average_latency_ms"] == 15.0
    assert snapshot["average_grounding_score"] == 0.75
    assert snapshot["average_retrieved_sources"] == 4.0
    assert snapshot["total_failed_agents"] == 1
