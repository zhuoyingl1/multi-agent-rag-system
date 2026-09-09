import json

import pytest

from multi_agent_rag.documents import _normalize_document_text, _normalize_extracted_text, load_document


def test_load_markdown_document(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nRAG grounds answers in evidence.", encoding="utf-8")

    document = load_document(path)

    assert document.title == "notes.md"
    assert document.metadata["document_type"] == "text"
    assert "RAG grounds" in document.text


def test_load_json_document_flattens_values(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"metrics": {"grounding": 0.9}, "tags": ["rag", "judge"]}), encoding="utf-8")

    document = load_document(path)

    assert document.metadata["document_type"] == "json"
    assert "metrics.grounding: 0.9" in document.text
    assert "tags[1]: judge" in document.text


def test_load_csv_document_serializes_rows(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("metric,value\nsource coverage,0.8\nlatency,120\n", encoding="utf-8")

    document = load_document(path)

    assert document.metadata["document_type"] == "csv"
    assert "Row 1: metric: source coverage; value: 0.8" in document.text
    assert "Row 2: metric: latency; value: 120" in document.text


def test_load_document_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "notes.docx"
    path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document extension"):
        load_document(path)


def test_normalize_extracted_text_cleans_pdf_artifacts() -> None:
    text = "Skills \uf06c RAG\u00a0systems\n\n\nTools \ufffd FastAPI"

    normalized = _normalize_extracted_text(text)

    assert "\uf06c" not in normalized
    assert "\ufffd" not in normalized
    assert "Skills - RAG systems" in normalized
    assert "\n\n\n" not in normalized


def test_normalize_document_text_removes_common_encoding_artifacts() -> None:
    text = "\ufeffRAG\u200b uses \ufb01ne-grained retrieval.\nPrivate\ue000 marker.\nBad\ufffd char."

    normalized = _normalize_document_text(text)

    assert "\ufeff" not in normalized
    assert "\u200b" not in normalized
    assert "\ue000" not in normalized
    assert "\ufffd" not in normalized
    assert "fine-grained retrieval" in normalized
    assert "Private marker." in normalized


def test_normalize_extracted_text_repairs_pdf_line_wrapping() -> None:
    text = "Retrieval aug-\nmented generation\nuses source evidence.\n\n- Grounding remains visible."

    normalized = _normalize_extracted_text(text)

    assert "augmented generation uses source evidence." in normalized
    assert "- Grounding remains visible." in normalized
    assert "aug-\nmented" not in normalized


def test_normalize_extracted_text_preserves_layout_boundaries() -> None:
    text = (
        "Projects\n"
        "First Project 01/2025-02/2025\n"
        "Tech Stack: Python\n"
        "- Built a service,\n"
        "then deployed it.\n"
        "Second Project 03/2025-04/2025\n"
        "Tech Stack: TypeScript"
    )

    normalized = _normalize_extracted_text(text)

    assert "Projects\n\nFirst Project" in normalized
    assert "deployed it.\n\nSecond Project" in normalized
    assert "- Built a service, then deployed it." in normalized


def test_load_text_document_applies_shared_normalization(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("RAG\u00a0systems use \ufb02exible retrieval.\ufffd", encoding="utf-8")

    document = load_document(path)

    assert "RAG systems use flexible retrieval." in document.text
    assert "\ufffd" not in document.text
