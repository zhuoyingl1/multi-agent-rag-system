"""Source-aware evidence formatting and citation diagnostics."""

from __future__ import annotations

import re
from typing import Any, Mapping

from multi_agent_rag.models import SearchResult


_CITATION_PATTERN = re.compile(r"\bS[1-9]\d*\b")


def citation_id(index: int) -> str:
    """Return the stable display id for a zero-based evidence position."""

    return f"S{index + 1}"


def format_evidence_context(sources: list[SearchResult], max_chars: int = 700) -> str:
    """Format ranked evidence with ids that an answer model can cite."""

    blocks: list[str] = []
    for index, source in enumerate(sources):
        title = source.chunk.metadata.get("title", source.chunk.document_id)
        location = source_locator(source)["label"]
        blocks.append(
            f"[{citation_id(index)}] Source: {title}; location: {location}; chunk: {source.chunk.index}; "
            f"type: {source.chunk.chunk_type.value}\n{_bounded_text(source.chunk.text, max_chars)}"
        )
    return "\n\n".join(blocks)


def source_locator(source: SearchResult) -> dict[str, int | str]:
    """Build a compact human-readable location from chunk metadata."""

    return build_source_locator(source.chunk.index, source.chunk.metadata)


def build_source_locator(chunk_index: int, metadata: Mapping[str, object]) -> dict[str, int | str]:
    """Build a source locator for retrieval results and stored chunk previews."""

    page_start = _optional_int(metadata.get("page_start"))
    page_end = _optional_int(metadata.get("page_end"))
    line_start = _optional_int(metadata.get("line_start"))
    line_end = _optional_int(metadata.get("line_end"))

    if page_start is not None:
        final_page = page_end or page_start
        label = f"page {page_start}" if final_page == page_start else f"pages {page_start}-{final_page}"
    elif line_start is not None:
        final_line = line_end or line_start
        label = f"line {line_start}" if final_line == line_start else f"lines {line_start}-{final_line}"
    else:
        label = f"chunk {chunk_index}"

    locator: dict[str, int | str] = {"label": label, "chunk_index": chunk_index}
    for key, value in (
        ("page_start", page_start),
        ("page_end", page_end),
        ("line_start", line_start),
        ("line_end", line_end),
    ):
        if value is not None:
            locator[key] = value
    return locator


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
        "source_locators": {
            citation_id(index): source_locator(source) for index, source in enumerate(sources)
        },
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


def _optional_int(value: object) -> int | None:
    try:
        return int(str(value)) if value is not None else None
    except ValueError:
        return None
