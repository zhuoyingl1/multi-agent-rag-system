"""Document ingestion utilities for local files."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from multi_agent_rag.models import Document

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
JSON_EXTENSIONS = {".json"}
CSV_EXTENSIONS = {".csv"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | JSON_EXTENSIONS | CSV_EXTENSIONS | PDF_EXTENSIONS
PDF_PAGE_BREAK_MARKER = "<!-- rag-page-break -->"


def load_document(path: str | Path) -> Document:
    """Load a local document into normalized text and metadata."""
    document_path = Path(path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")
    if not document_path.is_file():
        raise ValueError(f"Document path is not a file: {document_path}")

    extension = document_path.suffix.lower()
    if extension in TEXT_EXTENSIONS:
        text = document_path.read_text(encoding="utf-8")
        document_type = "text"
    elif extension in JSON_EXTENSIONS:
        text = _read_json(document_path)
        document_type = "json"
    elif extension in CSV_EXTENSIONS:
        text = _read_csv(document_path)
        document_type = "csv"
    elif extension in PDF_EXTENSIONS:
        text, page_count = _read_pdf(document_path)
        document_type = "pdf"
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported document extension '{extension}'. Supported extensions: {supported}")

    metadata = {
        "source_path": str(document_path),
        "document_type": document_type,
        "extension": extension,
    }
    if extension in PDF_EXTENSIONS:
        metadata["page_count"] = str(page_count)

    return Document(
        title=document_path.name,
        text=_normalize_document_text(text),
        metadata=metadata,
    )


def _read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = list(_flatten_json(data))
    return "\n".join(lines)


def _flatten_json(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(item, next_prefix))
        return lines
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            lines.extend(_flatten_json(item, next_prefix))
        return lines
    label = prefix or "value"
    return [f"{label}: {value}"]


def _read_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            rows = []
            for index, row in enumerate(reader, start=1):
                cells = [f"{key}: {value}" for key, value in row.items()]
                rows.append(f"Row {index}: " + "; ".join(cells))
            return "\n".join(rows)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return "\n".join(", ".join(cell for cell in row) for row in reader)


def _read_pdf(path: Path) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Install project dependencies before loading PDF files.") from exc

    reader = PdfReader(str(path))
    pages = [_normalize_extracted_text(page.extract_text() or "") for page in reader.pages]
    if not any(pages):
        raise ValueError(f"No extractable text found in PDF: {path}")
    return f"\n\n{PDF_PAGE_BREAK_MARKER}\n\n".join(pages), len(pages)


def _normalize_extracted_text(text: str) -> str:
    text = _normalize_document_text(text)
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    lines = text.splitlines()
    structured: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if structured and structured[-1]:
                structured.append("")
            continue

        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if _looks_like_pdf_boundary(stripped, next_line):
            if structured and structured[-1]:
                structured.append("")
            structured.append(stripped)
            continue

        if structured and structured[-1] and _is_pdf_continuation(structured[-1], stripped):
            structured[-1] = f"{structured[-1]} {stripped}"
        else:
            structured.append(stripped)
    return _collapse_blank_lines("\n".join(structured))


def _looks_like_pdf_boundary(line: str, next_line: str) -> bool:
    if line.startswith(("- ", "* ", "|", "#")):
        return False
    if line[0].islower():
        return False
    if re.search(r"\b\d{2}/\d{4}\s*-\s*\d{2}/\d{4}\b", line):
        return True
    next_starts_lowercase = bool(next_line) and next_line[0].islower()
    return len(line) <= 40 and len(line.split()) <= 5 and not next_starts_lowercase and not line.endswith((",", ";", ":"))


def _is_pdf_continuation(previous: str, current: str) -> bool:
    if current.startswith(("- ", "* ", "|", "#")):
        return False
    return current[0].islower() or previous.endswith((",", ";", "-"))


def _normalize_document_text(text: str) -> str:
    replacements = {
        "\ufeff": "",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2060": "",
        "\uf020": " ",
        "\uf06c": "- ",
        "\uf0b7": "- ",
        "\u2022": "- ",
        "\u25cf": "- ",
        "\ufffd": "",
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = "".join(_clean_character(character) for character in unicodedata.normalize("NFKC", text))
    return _collapse_blank_lines(text)


def _clean_character(character: str) -> str:
    if character in {"\n", "\t"}:
        return character
    category = unicodedata.category(character)
    if category.startswith("C"):
        return ""
    if category == "Co":
        return ""
    return character


def _collapse_blank_lines(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized_lines: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                normalized_lines.append("")
            blank_seen = True
            continue
        normalized_lines.append(line)
        blank_seen = False
    return "\n".join(normalized_lines).strip()
