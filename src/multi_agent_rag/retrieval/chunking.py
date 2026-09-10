"""Markdown-aware structured chunking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re

from multi_agent_rag.documents import PDF_PAGE_BREAK_MARKER, PPTX_SLIDE_BREAK_MARKER
from multi_agent_rag.models import Chunk, ChunkType, Document, stable_chunk_id


@dataclass(frozen=True)
class LocatedBlock:
    """A structured text block with its original line and source location."""

    chunk_type: ChunkType
    text: str
    line_start: int
    line_end: int
    location_start: int | None = None
    location_end: int | None = None


class StructuredChunker:
    """Split documents into prose, code, formula, and table chunks."""

    def __init__(self, max_prose_chars: int = 900, prose_overlap_chars: int = 220) -> None:
        self.max_prose_chars = max(200, max_prose_chars)
        self.prose_overlap_chars = min(max(0, prose_overlap_chars), self.max_prose_chars // 2)

    def chunk(self, document: Document) -> list[Chunk]:
        document_id = document.stable_id()
        document_type = document.metadata.get("document_type")
        location_kind = "page" if document_type == "pdf" else "slide" if document_type == "presentation" else None
        break_marker = PDF_PAGE_BREAK_MARKER if location_kind == "page" else PPTX_SLIDE_BREAK_MARKER if location_kind else None
        blocks = list(self._blocks(document.text, break_marker=break_marker))
        chunks: list[Chunk] = []
        for block in blocks:
            for located in self._split_block(block):
                index = len(chunks)
                metadata = {"title": document.title}
                metadata.update(document.metadata)
                metadata.update(
                    {
                        "line_start": str(located.line_start),
                        "line_end": str(located.line_end),
                    }
                )
                if located.location_start is not None and location_kind:
                    metadata[f"{location_kind}_start"] = str(located.location_start)
                    metadata[f"{location_kind}_end"] = str(located.location_end or located.location_start)
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_id=stable_chunk_id(document_id, index, located.chunk_type, located.text),
                        text=located.text,
                        chunk_type=located.chunk_type,
                        index=index,
                        metadata=metadata,
                    )
                )
        return chunks

    def _blocks(self, text: str, *, break_marker: str | None = None) -> Iterable[LocatedBlock]:
        lines = text.splitlines()
        prose_buffer: list[str] = []
        prose_start = 0
        index = 0
        location_index = 1

        def flush_prose() -> LocatedBlock | None:
            if not prose_buffer:
                return None
            value = "\n".join(prose_buffer).strip()
            line_start = prose_start
            line_end = line_start + len(prose_buffer) - 1
            prose_buffer.clear()
            if not value:
                return None
            location = location_index if break_marker else None
            return LocatedBlock(ChunkType.PROSE, value, line_start, line_end, location, location)

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            if break_marker and stripped == break_marker:
                pending = flush_prose()
                if pending:
                    yield pending
                location_index += 1
                index += 1
                continue

            if stripped.startswith("```"):
                pending = flush_prose()
                if pending:
                    yield pending
                start = index + 1
                code_lines = [line]
                index += 1
                while index < len(lines):
                    code_lines.append(lines[index])
                    if lines[index].strip().startswith("```"):
                        index += 1
                        break
                    index += 1
                yield LocatedBlock(
                    ChunkType.CODE, "\n".join(code_lines).strip(), start, index, location_index, location_index
                )
                continue

            if stripped.startswith("$$"):
                pending = flush_prose()
                if pending:
                    yield pending
                start = index + 1
                formula_lines = [line]
                index += 1
                while index < len(lines):
                    formula_lines.append(lines[index])
                    if lines[index].strip().endswith("$$"):
                        index += 1
                        break
                    index += 1
                yield LocatedBlock(
                    ChunkType.FORMULA, "\n".join(formula_lines).strip(), start, index, location_index, location_index
                )
                continue

            if self._is_table_line(line):
                pending = flush_prose()
                if pending:
                    yield pending
                start = index + 1
                table_lines = [line]
                index += 1
                while index < len(lines) and self._is_table_line(lines[index]):
                    table_lines.append(lines[index])
                    index += 1
                yield LocatedBlock(
                    ChunkType.TABLE, "\n".join(table_lines).strip(), start, index, location_index, location_index
                )
                continue

            if not stripped:
                pending = flush_prose()
                if pending:
                    yield pending
                index += 1
                continue

            if not prose_buffer:
                prose_start = index + 1
            prose_buffer.append(line)
            index += 1

        pending = flush_prose()
        if pending:
            yield pending

    def _split_block(self, block: LocatedBlock) -> Iterable[LocatedBlock]:
        if block.chunk_type is not ChunkType.PROSE or len(block.text) <= self.max_prose_chars:
            yield block
            return

        paragraph: list[re.Match[str]] = []
        current_length = 0
        for match in re.finditer(r"\S+", block.text):
            word = match.group()
            next_length = current_length + len(word) + (1 if paragraph else 0)
            if paragraph and next_length > self.max_prose_chars:
                yield self._located_piece(block, paragraph)
                paragraph = [*self._overlap_words(paragraph), match]
                current_length = sum(len(item.group()) for item in paragraph) + len(paragraph) - 1
            else:
                paragraph.append(match)
                current_length = next_length
        if paragraph:
            yield self._located_piece(block, paragraph)

    def _located_piece(self, block: LocatedBlock, words: list[re.Match[str]]) -> LocatedBlock:
        line_start = block.line_start + block.text.count("\n", 0, words[0].start())
        line_end = block.line_start + block.text.count("\n", 0, words[-1].end())
        return LocatedBlock(
            chunk_type=block.chunk_type,
            text=" ".join(word.group() for word in words),
            line_start=line_start,
            line_end=line_end,
            location_start=block.location_start,
            location_end=block.location_end,
        )

    def _overlap_words(self, words: list[re.Match[str]]) -> list[re.Match[str]]:
        selected: list[re.Match[str]] = []
        length = 0
        for word in reversed(words):
            next_length = length + len(word.group()) + (1 if selected else 0)
            if selected and next_length > self.prose_overlap_chars:
                break
            selected.append(word)
            length = next_length
        return list(reversed(selected))

    def _is_table_line(self, line: str) -> bool:
        stripped = line.strip()
        if "|" not in stripped:
            return False
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        return len(cells) >= 2 and any(cells)


def chunk_document(document: Document, max_prose_chars: int = 900) -> list[Chunk]:
    return StructuredChunker(max_prose_chars=max_prose_chars).chunk(document)
