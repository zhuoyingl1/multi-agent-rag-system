"""Persistent document vector indexing backed by Qdrant."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
from typing import Any, Protocol
import uuid

from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.retrieval.tokenization import tokenize


class EmbeddingProvider(Protocol):
    """Minimal batch embedding contract used by the vector index."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class QdrantDocumentIndex:
    """Replace and persist all vector points belonging to one document."""

    def __init__(
        self,
        url: str,
        collection: str,
        embedder: EmbeddingProvider,
        batch_size: int = 50,
        score_threshold: float = 0.5,
        client: Any | None = None,
    ) -> None:
        self.url = url
        self.collection = collection
        self.embedder = embedder
        self.batch_size = max(1, batch_size)
        self.score_threshold = score_threshold
        self._client_injected = client is not None
        self.client = client or self._build_client(url)

    def replace_document_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> int:
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("Every indexed chunk must belong to the requested document.")
        if not chunks:
            self._delete_document(document_id)
            return 0

        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors.extend(self.embedder.encode([chunk.text for chunk in batch]))

        vector_size = self._validate_vectors(vectors, len(chunks))
        self._ensure_collection(vector_size)
        self._delete_document(document_id)

        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start : start + self.batch_size]
            batch_vectors = vectors[start : start + self.batch_size]
            points = [self._point(chunk, vector) for chunk, vector in zip(batch_chunks, batch_vectors)]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(chunks)

    def search(self, document_id: str, query: str, limit: int) -> list[SearchResult]:
        if not self.client.collection_exists(self.collection):
            return []
        vectors = self.embedder.encode([query])
        self._validate_vectors(vectors, 1)
        response = self.client.query_points(
            collection_name=self.collection,
            query=vectors[0],
            query_filter=self._document_filter(document_id),
            limit=limit,
            score_threshold=self.score_threshold,
            with_payload=True,
        )
        points = list(getattr(response, "points", response))
        terms = tokenize(query)
        return [self._search_result(point, terms) for point in points]

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()

    def _build_client(self, url: str) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Qdrant indexing requires qdrant-client.") from exc
        return QdrantClient(url=url)

    def _validate_vectors(self, vectors: list[list[float]], expected_count: int) -> int:
        if len(vectors) != expected_count:
            raise RuntimeError(f"Embedding provider returned {len(vectors)} vectors for {expected_count} chunks.")
        vector_size = len(vectors[0])
        if vector_size == 0:
            raise RuntimeError("Embedding vectors must not be empty.")
        if any(len(vector) != vector_size for vector in vectors):
            raise RuntimeError("Embedding vectors must all have the same dimension.")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise RuntimeError("Embedding vectors must contain only finite values.")
        return vector_size

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            if not self._client_injected:
                raise RuntimeError("Qdrant collection setup requires qdrant-client models.") from exc
            vectors_config: Any = {"size": vector_size, "distance": "Cosine"}
        else:
            vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)
        self.client.create_collection(collection_name=self.collection, vectors_config=vectors_config)

    def _delete_document(self, document_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=self._document_selector(document_id),
            wait=True,
        )

    def _document_filter(self, document_id: str) -> Any:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError:
            return {"must": [{"key": "document_id", "match": {"value": document_id}}]}
        return Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])

    def _document_selector(self, document_id: str) -> Any:
        document_filter = self._document_filter(document_id)
        try:
            from qdrant_client.models import FilterSelector
        except ImportError:
            return {"filter": document_filter}
        return FilterSelector(filter=document_filter)

    def _point(self, chunk: Chunk, vector: list[float]) -> Any:
        point_id = str(uuid.UUID(hashlib.sha256(chunk.chunk_id.encode("utf-8")).hexdigest()[:32]))
        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "chunk_type": chunk.chunk_type.value,
            "chunk_index": chunk.index,
            "metadata": chunk.metadata,
        }
        if self._client_injected:
            return {"id": point_id, "vector": vector, "payload": payload}
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            return {"id": point_id, "vector": vector, "payload": payload}
        return PointStruct(id=point_id, vector=vector, payload=payload)

    def _search_result(self, point: Any, terms: list[str]) -> SearchResult:
        payload = dict(point.get("payload", {}) if isinstance(point, dict) else getattr(point, "payload", {}))
        chunk = Chunk(
            document_id=str(payload["document_id"]),
            chunk_id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            chunk_type=ChunkType(str(payload["chunk_type"])),
            index=int(payload["chunk_index"]),
            metadata={str(key): str(value) for key, value in dict(payload.get("metadata", {})).items()},
        )
        score = float(point.get("score", 0.0) if isinstance(point, dict) else getattr(point, "score", 0.0))
        lowered = chunk.text.lower()
        highlights = [term for term in terms if term.lower() in lowered]
        return SearchResult(
            chunk=chunk,
            score=score,
            retrieval_type=RetrievalType.VECTOR,
            highlights=list(dict.fromkeys(highlights))[:5],
        )


class QdrantDocumentRetriever:
    """Retrieve only from vectors previously indexed for one document."""

    def __init__(self, index: QdrantDocumentIndex, document_id: str, top_k: int = 5) -> None:
        self.index = index
        self.document_id = document_id
        self.top_k = top_k

    def index(self, chunks: list[Chunk]) -> None:
        raise RuntimeError("Persistent document chunks must be indexed during ingestion.")

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        return self.index.search(self.document_id, query, top_k or self.top_k)

    def close(self) -> None:
        self.index.close()
