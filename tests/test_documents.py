import json

import pytest

from multi_agent_rag.documents import load_document


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
