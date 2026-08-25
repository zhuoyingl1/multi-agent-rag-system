"""Qdrant-backed vector retrieval adapter."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.retrieval.embeddings import VECTOR_SIZE, hashed_embedding
from multi_agent_rag.retrieval.tokenization import tokenize


class QdrantRetriever:
    """Store and search chunks through a Qdrant collection."""

    def __init__(
        self,
        url: str,
        collection: str,
        top_k: int = 5,
        vector_size: int = VECTOR_SIZE,
        session_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.url = url
        self.collection = collection
        self.top_k = top_k
        self.vector_size = vector_size
        self.session_id = session_id or uuid.uuid4().hex
        self._client_injected = client is not None
        self.client = client or self._build_client(url)
        self._ensure_collection()

    def index(self, chunks: list[Chunk]) -> None:
        points = [self._point(chunk) for chunk in chunks]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        limit = top_k or self.top_k
        vector = hashed_embedding(query, size=self.vector_size)
        points = self._search(vector, limit)
        terms = tokenize(query)
        return [self._result(point, terms) for point in points]

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()

    def _build_client(self, url: str) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Qdrant retrieval requires qdrant-client. Install the project dependencies first.") from exc
        return QdrantClient(url=url)

    def _ensure_collection(self) -> None:
        if hasattr(self.client, "collection_exists") and self.client.collection_exists(self.collection):
            return
        if not hasattr(self.client, "collection_exists"):
            try:
                self.client.get_collection(self.collection)
                return
            except Exception:
                pass

        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            if not self._client_injected:
                raise RuntimeError("Qdrant collection setup requires qdrant-client models.") from exc
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={"size": self.vector_size, "distance": "Cosine"},
            )
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def _point(self, chunk: Chunk) -> Any:
        payload = {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "chunk_type": chunk.chunk_type.value,
            "index": chunk.index,
            "metadata": chunk.metadata,
            "session_id": self.session_id,
        }
        point_id = str(uuid.UUID(hashlib.sha256(chunk.chunk_id.encode("utf-8")).hexdigest()[:32]))
        vector = hashed_embedding(chunk.text, size=self.vector_size)
        if self._client_injected:
            return {"id": point_id, "vector": vector, "payload": payload}
        try:
            from qdrant_client.models import PointStruct

            return PointStruct(id=point_id, vector=vector, payload=payload)
        except ImportError:
            return {"id": point_id, "vector": vector, "payload": payload}

    def _search(self, vector: list[float], limit: int) -> list[Any]:
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=limit,
                query_filter=self._session_filter(),
                with_payload=True,
            )
            return list(getattr(response, "points", response))
        if hasattr(self.client, "search"):
            return list(
                self.client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=limit,
                    query_filter=self._session_filter(),
                    with_payload=True,
                )
            )
        raise RuntimeError("Qdrant client does not expose a supported search method.")

    def _session_filter(self) -> Any:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            return Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=self.session_id))])
        except ImportError:
            return {"must": [{"key": "session_id", "match": {"value": self.session_id}}]}

    def _result(self, point: Any, terms: list[str]) -> SearchResult:
        payload = dict(point.get("payload", {}) if isinstance(point, dict) else getattr(point, "payload", {}))
        chunk = Chunk(
            document_id=str(payload["document_id"]),
            chunk_id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            chunk_type=ChunkType(str(payload["chunk_type"])),
            index=int(payload["index"]),
            metadata={str(key): str(value) for key, value in dict(payload.get("metadata", {})).items()},
        )
        text = chunk.text.lower()
        highlights = []
        for term in terms:
            lowered = term.lower()
            if lowered in text and lowered not in highlights:
                highlights.append(lowered)
        score = float(point.get("score", 0.0) if isinstance(point, dict) else getattr(point, "score", 0.0))
        return SearchResult(
            chunk=chunk,
            score=score,
            retrieval_type=RetrievalType.VECTOR,
            highlights=highlights[:5],
        )
