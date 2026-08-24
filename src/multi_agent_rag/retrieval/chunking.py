"""Markdown-aware structured chunking."""

from __future__ import annotations

from collections.abc import Iterable

from multi_agent_rag.models import Chunk, ChunkType, Document, stable_chunk_id


class StructuredChunker:
    """Split documents into prose, code, formula, and table chunks."""

    def __init__(self, max_prose_chars: int = 900) -> None:
        self.max_prose_chars = max(200, max_prose_chars)

    def chunk(self, document: Document) -> list[Chunk]:
        document_id = document.stable_id()
        blocks = list(self._blocks(document.text))
        chunks: list[Chunk] = []
        for block_type, block_text in blocks:
            for text in self._split_block(block_text, block_type):
                index = len(chunks)
                metadata = {"title": document.title}
                metadata.update(document.metadata)
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_id=stable_chunk_id(document_id, index, block_type, text),
                        text=text,
                        chunk_type=block_type,
                        index=index,
                        metadata=metadata,
                    )
                )
        return chunks

    def _blocks(self, text: str) -> Iterable[tuple[ChunkType, str]]:
        lines = text.splitlines()
        prose_buffer: list[str] = []
        index = 0

        def flush_prose() -> tuple[ChunkType, str] | None:
            if not prose_buffer:
                return None
            value = "\n".join(prose_buffer).strip()
            prose_buffer.clear()
            if not value:
                return None
            return ChunkType.PROSE, value

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            if stripped.startswith("```"):
                pending = flush_prose()
                if pending:
                    yield pending
                code_lines = [line]
                index += 1
                while index < len(lines):
                    code_lines.append(lines[index])
                    if lines[index].strip().startswith("```"):
                        index += 1
                        break
                    index += 1
                yield ChunkType.CODE, "\n".join(code_lines).strip()
                continue

            if stripped.startswith("$$"):
                pending = flush_prose()
                if pending:
                    yield pending
                formula_lines = [line]
                index += 1
                while index < len(lines):
                    formula_lines.append(lines[index])
                    if lines[index].strip().endswith("$$"):
                        index += 1
                        break
                    index += 1
                yield ChunkType.FORMULA, "\n".join(formula_lines).strip()
                continue

            if self._is_table_line(line):
                pending = flush_prose()
                if pending:
                    yield pending
                table_lines = [line]
                index += 1
                while index < len(lines) and self._is_table_line(lines[index]):
                    table_lines.append(lines[index])
                    index += 1
                yield ChunkType.TABLE, "\n".join(table_lines).strip()
                continue

            if not stripped:
                pending = flush_prose()
                if pending:
                    yield pending
                index += 1
                continue

            prose_buffer.append(line)
            index += 1

        pending = flush_prose()
        if pending:
            yield pending

    def _split_block(self, text: str, chunk_type: ChunkType) -> Iterable[str]:
        if chunk_type is not ChunkType.PROSE or len(text) <= self.max_prose_chars:
            yield text
            return

        paragraph = []
        current_length = 0
        for word in text.split():
            next_length = current_length + len(word) + (1 if paragraph else 0)
            if paragraph and next_length > self.max_prose_chars:
                yield " ".join(paragraph)
                paragraph = [word]
                current_length = len(word)
            else:
                paragraph.append(word)
                current_length = next_length
        if paragraph:
            yield " ".join(paragraph)

    def _is_table_line(self, line: str) -> bool:
        stripped = line.strip()
        if "|" not in stripped:
            return False
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        return len(cells) >= 2 and any(cells)


def chunk_document(document: Document, max_prose_chars: int = 900) -> list[Chunk]:
    return StructuredChunker(max_prose_chars=max_prose_chars).chunk(document)
