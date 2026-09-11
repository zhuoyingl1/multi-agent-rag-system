import pytest

from multi_agent_rag.quality_gates import evaluate_quality_gates


def test_quality_gates_report_minimum_and_maximum_results() -> None:
    result = evaluate_quality_gates(
        {"pass_rate": 0.9, "latency_ms": 120.0},
        minimums={"pass_rate": 0.8},
        maximums={"latency_ms": 100.0},
    )

    assert result.passed is False
    assert [check.passed for check in result.checks] == [True, False]
    assert result.to_dict()["checks"][1]["metric"] == "latency_ms"


def test_quality_gates_allow_no_configured_thresholds() -> None:
    result = evaluate_quality_gates({"pass_rate": 0.0})

    assert result.passed is True
    assert result.checks == []


def test_quality_gates_reject_unknown_metrics() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        evaluate_quality_gates({"pass_rate": 1.0}, minimums={"recall": 0.8})


def test_quality_gates_reject_negative_thresholds() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluate_quality_gates({"latency_ms": 10.0}, maximums={"latency_ms": -1.0})
