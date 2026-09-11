"""Reusable metric thresholds for evaluation quality gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class QualityGateCheck:
    """One minimum or maximum threshold comparison."""

    metric: str
    comparison: str
    threshold: float
    actual: float
    passed: bool


@dataclass(frozen=True)
class QualityGateResult:
    """Combined result for all configured metric thresholds."""

    passed: bool
    checks: list[QualityGateCheck]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_quality_gates(
    metrics: Mapping[str, float],
    *,
    minimums: Mapping[str, float | None] | None = None,
    maximums: Mapping[str, float | None] | None = None,
) -> QualityGateResult:
    """Compare metrics with configured bounds and return every check."""

    checks: list[QualityGateCheck] = []
    for metric, threshold in (minimums or {}).items():
        if threshold is not None:
            checks.append(_check(metrics, metric, "minimum", threshold))
    for metric, threshold in (maximums or {}).items():
        if threshold is not None:
            checks.append(_check(metrics, metric, "maximum", threshold))
    return QualityGateResult(passed=all(check.passed for check in checks), checks=checks)


def _check(
    metrics: Mapping[str, float],
    metric: str,
    comparison: str,
    threshold: float,
) -> QualityGateCheck:
    if metric not in metrics:
        raise ValueError(f"Quality gate metric is unavailable: {metric}")
    actual = float(metrics[metric])
    threshold = float(threshold)
    if not math.isfinite(actual) or not math.isfinite(threshold):
        raise ValueError(f"Quality gate values must be finite for {metric}.")
    if threshold < 0:
        raise ValueError(f"Quality gate threshold cannot be negative for {metric}.")
    passed = actual >= threshold if comparison == "minimum" else actual <= threshold
    return QualityGateCheck(
        metric=metric,
        comparison=comparison,
        threshold=threshold,
        actual=actual,
        passed=passed,
    )
