"""Embedding providers used by local and persistent retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multi_agent_rag.retrieval.tokenization import token_counts

VECTOR_SIZE = 64


class OllamaEmbeddingService:
    """Generate real text embeddings through Ollama's local HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "nomic-embed-text",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embedding request failed with HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama embedding service is unavailable at {self.base_url}: {exc}") from exc

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response.")
        return [[float(value) for value in vector] for vector in embeddings]


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
