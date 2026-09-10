from multi_agent_rag.documents import PPTX_SLIDE_BREAK_MARKER
from multi_agent_rag.models import ChunkType, Document
from multi_agent_rag.retrieval.chunking import StructuredChunker, chunk_document


SAMPLE_TEXT = """# Research Notes

Retrieval augmented generation grounds generated answers in source evidence.

```python
def score(answer: str) -> float:
    return 1.0
```

| Metric | Meaning |
| --- | --- |
| Source coverage | Share of answer claims with evidence |

$$
precision = relevant / retrieved
$$
"""


def test_chunk_document_detects_structured_blocks() -> None:
    document = Document(title="notes.md", text=SAMPLE_TEXT, metadata={"source": "example"})

    chunks = chunk_document(document)
    chunk_types = {chunk.chunk_type for chunk in chunks}

    assert ChunkType.PROSE in chunk_types
    assert ChunkType.CODE in chunk_types
    assert ChunkType.TABLE in chunk_types
    assert ChunkType.FORMULA in chunk_types
    assert all(chunk.document_id == document.stable_id() for chunk in chunks)
    assert all(chunk.metadata["title"] == "notes.md" for chunk in chunks)
    assert all(chunk.metadata["source"] == "example" for chunk in chunks)
    assert all(int(chunk.metadata["line_start"]) <= int(chunk.metadata["line_end"]) for chunk in chunks)


def test_chunk_document_tracks_pdf_pages_without_indexing_markers() -> None:
    document = Document(
        title="report.pdf",
        text="First page.\n\n<!-- rag-page-break -->\n\nSecond page.",
        metadata={"document_type": "pdf", "page_count": "2"},
    )

    chunks = chunk_document(document)

    assert [chunk.metadata["page_start"] for chunk in chunks] == ["1", "2"]
    assert all("rag-page-break" not in chunk.text for chunk in chunks)


def test_chunk_document_tracks_presentation_slides_without_indexing_markers() -> None:
    document = Document(
        title="briefing.pptx",
        text=f"First slide.\n\n{PPTX_SLIDE_BREAK_MARKER}\n\nSecond slide.",
        metadata={"document_type": "presentation", "slide_count": "2"},
    )

    chunks = chunk_document(document)

    assert [chunk.metadata["slide_start"] for chunk in chunks] == ["1", "2"]
    assert all("rag-slide-break" not in chunk.text for chunk in chunks)


def test_chunk_ids_are_stable() -> None:
    document = Document(title="notes.md", text=SAMPLE_TEXT)

    first = chunk_document(document)
    second = chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_document_tracks_markdown_sections() -> None:
    document = Document(title="notes.md", text="# Retrieval\n\nEvidence.\n\n# Evaluation\n\nMetrics.")

    chunks = chunk_document(document)

    sections = list(dict.fromkeys(chunk.metadata["section"] for chunk in chunks))

    assert sections == ["Retrieval", "Evaluation"]


def test_long_prose_is_split_without_losing_words() -> None:
    text = " ".join(f"word{i}" for i in range(120))
    document = Document(title="long.txt", text=text)

    chunks = StructuredChunker(max_prose_chars=240).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.chunk_type is ChunkType.PROSE for chunk in chunks)
    assert "word0" in chunks[0].text
    assert "word119" in chunks[-1].text
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())


def test_split_prose_keeps_the_original_line_range() -> None:
    text = "\n".join(" ".join(f"line{line}word{word}" for word in range(20)) for line in range(1, 4))
    chunks = StructuredChunker(max_prose_chars=240).chunk(Document(title="long.txt", text=text))

    assert len(chunks) > 1
    assert chunks[0].metadata["line_start"] == "1"
    assert chunks[-1].metadata["line_end"] == "3"
