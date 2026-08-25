"""Neo4j graph adapter for chunk and entity relationships."""

from __future__ import annotations

from typing import Any

from multi_agent_rag.models import Chunk
from multi_agent_rag.retrieval.tokenization import extract_entities


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
        self.driver = driver or self._build_driver(uri, user, password)
        self._ensure_constraints()

    def index(self, chunks: list[Chunk]) -> None:
        with self.driver.session(database=self.database) as session:
            for chunk in chunks:
                entities = sorted(extract_entities(chunk.text))
                session.execute_write(self._merge_chunk, chunk, entities)

    def expand_entities(self, query: str, limit: int = 8) -> list[str]:
        entities = sorted(extract_entities(query))
        if not entities:
            return []
        with self.driver.session(database=self.database) as session:
            records = session.execute_read(self._related_entities, entities, limit)
        return [str(record["entity"]) for record in records]

    def close(self) -> None:
        self.driver.close()

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
