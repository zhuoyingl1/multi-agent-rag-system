"""Neo4j graph adapter for chunk and entity relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.persistence import ChunkRepository
from multi_agent_rag.retrieval.tokenization import extract_entities, tokenize


@dataclass(frozen=True)
class GraphChunkMatch:
    """A document chunk reached through matching graph entities."""

    chunk_id: str
    score: float
    entities: list[str]


class Neo4jGraphAdapter:
    """Persist chunk/entity relationships and expand query entities."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        driver: Any | None = None,
    ) -> None:
        self.uri = uri
        self.user = user
        self.database = database
        self.password = password
        self.driver = driver
        if self.driver is not None:
            self._ensure_constraints()

    def index(self, chunks: list[Chunk]) -> None:
        with self._get_driver().session(database=self.database) as session:
            for chunk in chunks:
                entities = sorted(extract_entities(chunk.text))
                session.execute_write(self._merge_chunk, chunk, entities)

    def expand_entities(self, query: str, limit: int = 8) -> list[str]:
        entities = sorted(extract_entities(query))
        if not entities:
            return []
        with self._get_driver().session(database=self.database) as session:
            records = session.execute_read(self._related_entities, entities, limit)
        return [str(record["entity"]) for record in records]

    def retrieve_chunk_matches(self, query: str, document_id: str, limit: int = 50) -> list[GraphChunkMatch]:
        """Find chunks connected to query entities inside one document graph."""

        entities = sorted(extract_entities(query) | {term for term in tokenize(query) if len(term) > 2})
        if not entities:
            return []
        with self._get_driver().session(database=self.database) as session:
            records = session.execute_read(self._related_chunks, entities, document_id, limit)
        return [
            GraphChunkMatch(
                chunk_id=str(record["chunk_id"]),
                score=float(record["score"]),
                entities=[str(entity) for entity in record.get("entities", [])],
            )
            for record in records
        ]

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
        self.driver = None

    def _get_driver(self) -> Any:
        if self.driver is None:
            self.driver = self._build_driver(self.uri, self.user, self.password)
            self._ensure_constraints()
        return self.driver

    def _build_driver(self, uri: str, user: str, password: str) -> Any:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j graph retrieval requires the neo4j package. Install project dependencies first.") from exc
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver

    def _ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT rag_document_id IF NOT EXISTS FOR (d:RagDocument) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT rag_chunk_id IF NOT EXISTS FOR (c:RagChunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT rag_entity_name IF NOT EXISTS FOR (e:RagEntity) REQUIRE e.name IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    @staticmethod
    def _merge_chunk(tx: Any, chunk: Chunk, entities: list[str]) -> None:
        tx.run(
            """
            MERGE (d:RagDocument {id: $document_id})
            SET d.title = $title
            MERGE (c:RagChunk {id: $chunk_id})
            SET c.text = $text,
                c.chunk_type = $chunk_type,
                c.index = $index,
                c.title = $title
            MERGE (d)-[:HAS_CHUNK]->(c)
            WITH c
            UNWIND $entities AS entity_name
            MERGE (e:RagEntity {name: entity_name})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            chunk_type=chunk.chunk_type.value,
            index=chunk.index,
            title=chunk.metadata.get("title", chunk.document_id),
            entities=entities,
        )

    @staticmethod
    def _related_entities(tx: Any, entities: list[str], limit: int) -> list[Any]:
        result = tx.run(
            """
            UNWIND $entities AS query_entity
            MATCH (:RagEntity {name: query_entity})<-[:MENTIONS]-(:RagChunk)-[:MENTIONS]->(related:RagEntity)
            WHERE NOT related.name IN $entities
            RETURN related.name AS entity, count(*) AS support
            ORDER BY support DESC, entity ASC
            LIMIT $limit
            """,
            entities=entities,
            limit=limit,
        )
        return list(result)

    @staticmethod
    def _related_chunks(tx: Any, entities: list[str], document_id: str, limit: int) -> list[Any]:
        result = tx.run(
            """
            UNWIND $entities AS query_entity
            MATCH (document:RagDocument {id: $document_id})-[:HAS_CHUNK]->(seed:RagChunk)
            MATCH (seed)-[:MENTIONS]->(:RagEntity {name: query_entity})
            OPTIONAL MATCH (seed)-[:MENTIONS]->(shared:RagEntity)<-[:MENTIONS]-(related:RagChunk)<-[:HAS_CHUNK]-(document)
            WITH seed, query_entity, collect(DISTINCT related) AS related_chunks
            UNWIND [seed] + related_chunks AS candidate
            WITH candidate, collect(DISTINCT query_entity) AS matched_entities
            WHERE candidate IS NOT NULL
            RETURN candidate.id AS chunk_id,
                   toFloat(size(matched_entities)) AS score,
                   matched_entities AS entities
            ORDER BY score DESC, chunk_id ASC
            LIMIT $limit
            """,
            entities=entities,
            document_id=document_id,
            limit=limit,
        )
        return list(result)


class Neo4jGraphRetriever:
    """Retrieve persisted chunks through document-scoped graph relationships."""

    def __init__(
        self,
        adapter: Neo4jGraphAdapter,
        chunks: ChunkRepository,
        document_id: str,
        top_k: int = 50,
    ) -> None:
        self.adapter = adapter
        self.chunks = chunks
        self.document_id = document_id
        self.top_k = top_k

    def index(self, chunks: list[Chunk]) -> None:
        raise RuntimeError("Persistent document chunks must be indexed during ingestion.")

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        matches = self.adapter.retrieve_chunk_matches(query, self.document_id, top_k or self.top_k)
        if not matches:
            return []

        records = {record.chunk_id: record for record in self.chunks.list_for_document(self.document_id)}
        results: list[SearchResult] = []
        for match in matches:
            record = records.get(match.chunk_id)
            if record is None:
                continue
            chunk = Chunk(
                document_id=record.document_id,
                chunk_id=record.chunk_id,
                text=record.text,
                chunk_type=ChunkType(record.chunk_type),
                index=record.index,
                metadata={str(key): str(value) for key, value in record.metadata.items()},
            )
            results.append(SearchResult(chunk, match.score, RetrievalType.GRAPH, match.entities[:5]))
        return results

    def close(self) -> None:
        self.adapter.close()
