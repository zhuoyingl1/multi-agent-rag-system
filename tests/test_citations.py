from multi_agent_rag.citations import build_citation_diagnostics, build_source_locator, extract_citation_ids, format_evidence_context
from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult


def source(index: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            document_id="doc-1",
            chunk_id=f"chunk-{index}",
            text=f"Evidence statement {index}.",
            chunk_type=ChunkType.PROSE,
            index=index,
            metadata={"title": "notes.md", "line_start": str(index + 4), "line_end": str(index + 5)},
        ),
        score=1.0 - index * 0.1,
        retrieval_type=RetrievalType.HYBRID,
    )


def test_evidence_context_assigns_ranked_citation_ids() -> None:
    context = format_evidence_context([source(0), source(1)])

    assert "[S1] Source: notes.md; location: lines 4-5; chunk: 0" in context
    assert "[S2] Source: notes.md; location: lines 5-6; chunk: 1" in context


def test_extract_citation_ids_deduplicates_in_first_use_order() -> None:
    assert extract_citation_ids("Use [S2], then [S1], and repeat [S2].") == ["S2", "S1"]
    assert extract_citation_ids("Combined evidence [S1, S2].") == ["S1", "S2"]


def test_citation_diagnostics_reports_complete_and_partial_answers() -> None:
    sources = [source(0), source(1)]

    complete = build_citation_diagnostics("First [S1]. Second [S2].", sources)
    partial = build_citation_diagnostics("Only first [S1].", sources)

    assert complete["status"] == "complete"
    assert complete["coverage"] == 1.0
    assert complete["source_locators"]["S1"]["label"] == "lines 4-5"
    assert partial["status"] == "partial"
    assert partial["unused_citation_ids"] == ["S2"]


def test_citation_diagnostics_flags_missing_and_invalid_ids() -> None:
    sources = [source(0)]

    missing = build_citation_diagnostics("No citation is present.", sources)
    invalid = build_citation_diagnostics("Unsupported reference [S9].", sources)

    assert missing["status"] == "missing"
    assert invalid["status"] == "invalid"
    assert invalid["invalid_citation_ids"] == ["S9"]


def test_source_locator_can_be_reused_for_stored_chunk_preview() -> None:
    locator = build_source_locator(3, {"page_start": "2", "page_end": "4"})

    assert locator == {
        "label": "pages 2-4",
        "chunk_index": 3,
        "page_start": 2,
        "page_end": 4,
    }


def test_source_locator_formats_presentation_slides() -> None:
    locator = build_source_locator(2, {"slide_start": "3", "slide_end": "5"})

    assert locator == {
        "label": "slides 3-5",
        "chunk_index": 2,
        "slide_start": 3,
        "slide_end": 5,
    }
