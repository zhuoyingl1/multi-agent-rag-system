"""Source-aware evidence formatting and citation diagnostics."""

from __future__ import annotations

import re
from typing import Any

from multi_agent_rag.models import SearchResult


_CITATION_PATTERN = re.compile(r"\[(S[1-9]\d*)\]")


def citation_id(index: int) -> str:
    """Return the stable display id for a zero-based evidence position."""

    return f"S{index + 1}"


def format_evidence_context(sources: list[SearchResult], max_chars: int = 700) -> str:
    """Format ranked evidence with ids that an answer model can cite."""

    blocks: list[str] = []
    for index, source in enumerate(sources):
        title = source.chunk.metadata.get("title", source.chunk.document_id)
        blocks.append(
            f"[{citation_id(index)}] Source: {title}; chunk: {source.chunk.index}; "
            f"type: {source.chunk.chunk_type.value}\n{_bounded_text(source.chunk.text, max_chars)}"
        )
    return "\n\n".join(blocks)


def extract_citation_ids(answer: str) -> list[str]:
    """Extract distinct bracketed citation ids in first-use order."""

    return list(dict.fromkeys(_CITATION_PATTERN.findall(answer or "")))


def build_citation_diagnostics(answer: str, sources: list[SearchResult]) -> dict[str, Any]:
    """Describe whether an answer cites only the retrieved evidence ids."""

    available_ids = [citation_id(index) for index in range(len(sources))]
    available = set(available_ids)
    used_ids = extract_citation_ids(answer)
    valid_ids = [item for item in used_ids if item in available]
    invalid_ids = [item for item in used_ids if item not in available]
    unused_ids = [item for item in available_ids if item not in set(valid_ids)]

    if not available_ids:
        status = "no_evidence"
    elif invalid_ids:
        status = "invalid"
    elif not valid_ids:
        status = "missing"
    elif not unused_ids:
        status = "complete"
    else:
        status = "partial"

    coverage = len(valid_ids) / len(available_ids) if available_ids else 0.0
    return {
        "status": status,
        "evidence_count": len(available_ids),
        "cited_source_count": len(valid_ids),
        "used_citation_ids": used_ids,
        "valid_citation_ids": valid_ids,
        "invalid_citation_ids": invalid_ids,
        "unused_citation_ids": unused_ids,
        "coverage": round(coverage, 4),
    }


def citation_metrics(answer: str, sources: list[SearchResult]) -> dict[str, float | int | str]:
    """Return compact citation fields suitable for workflow metrics."""

    diagnostics = build_citation_diagnostics(answer, sources)
    return {
        "citation_status": str(diagnostics["status"]),
        "citation_coverage": float(diagnostics["coverage"]),
        "cited_sources": int(diagnostics["cited_source_count"]),
        "invalid_citations": len(diagnostics["invalid_citation_ids"]),
    }


def _bounded_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    boundary = compact.rfind(" ", 0, max_chars)
    end = boundary if boundary > max_chars // 2 else max_chars
    return compact[:end].rstrip() + "..."
