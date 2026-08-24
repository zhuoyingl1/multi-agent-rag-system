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


def test_chunk_ids_are_stable() -> None:
    document = Document(title="notes.md", text=SAMPLE_TEXT)

    first = chunk_document(document)
    second = chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_long_prose_is_split_without_losing_words() -> None:
    text = " ".join(f"word{i}" for i in range(120))
    document = Document(title="long.txt", text=text)

    chunks = StructuredChunker(max_prose_chars=240).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.chunk_type is ChunkType.PROSE for chunk in chunks)
    assert "word0" in chunks[0].text
    assert "word119" in chunks[-1].text
