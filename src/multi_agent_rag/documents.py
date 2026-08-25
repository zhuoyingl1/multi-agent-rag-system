"""Document ingestion utilities for local files."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from multi_agent_rag.models import Document

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
JSON_EXTENSIONS = {".json"}
CSV_EXTENSIONS = {".csv"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | JSON_EXTENSIONS | CSV_EXTENSIONS | PDF_EXTENSIONS


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
        text = _read_pdf(document_path)
        document_type = "pdf"
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported document extension '{extension}'. Supported extensions: {supported}")

    return Document(
        title=document_path.name,
        text=text,
        metadata={
            "source_path": str(document_path),
            "document_type": document_type,
            "extension": extension,
        },
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


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Install project dependencies before loading PDF files.") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = _normalize_extracted_text("\n\n".join(page for page in pages if page))
    if not text:
        raise ValueError(f"No extractable text found in PDF: {path}")
    return text


def _normalize_extracted_text(text: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\uf020": " ",
        "\uf06c": "- ",
        "\uf0b7": "- ",
        "\u2022": "- ",
        "\u25cf": "- ",
        "\ufffd": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

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
