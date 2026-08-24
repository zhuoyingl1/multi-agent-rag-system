"""Tokenization helpers for local retrieval."""

from __future__ import annotations

import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_+-]{2,}\b")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def token_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def extract_entities(text: str) -> set[str]:
    return {match.group(0).lower() for match in ENTITY_PATTERN.finditer(text)}
