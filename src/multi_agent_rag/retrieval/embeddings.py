"""Deterministic lightweight embeddings for local and adapter tests."""

from __future__ import annotations

import hashlib
import math

from multi_agent_rag.retrieval.tokenization import token_counts

VECTOR_SIZE = 64


def hashed_embedding(text: str, size: int = VECTOR_SIZE) -> list[float]:
    vector = [0.0] * size
    for token, count in token_counts(text).items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * float(count)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]
