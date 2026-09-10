"""Factories for document ingestion services shared by API and workers."""

from __future__ import annotations

import os

from multi_agent_rag.ingestion import DocumentIngestionService
from multi_agent_rag.persistence import ChunkRepository, ConversationRepository, DocumentRepository, MongoStore
from multi_agent_rag.retrieval.embeddings import OllamaEmbeddingService
from multi_agent_rag.retrieval.neo4j_adapter import Neo4jGraphAdapter
from multi_agent_rag.retrieval.vector_index import QdrantDocumentIndex


def create_document_ingestion_service(store: MongoStore) -> DocumentIngestionService:
    """Build the persistent ingestion pipeline around a shared MongoDB store."""

    embedder = OllamaEmbeddingService(
        base_url=os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434",
        model_name=os.getenv("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text",
        timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS") or "60"),
    )
    vector_index = QdrantDocumentIndex(
        url=os.getenv("QDRANT_URL") or "http://localhost:6333",
        collection=os.getenv("QDRANT_DOCUMENT_COLLECTION") or "document_chunks",
        embedder=embedder,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE") or "50"),
    )
    graph_index = Neo4jGraphAdapter(
        uri=os.getenv("NEO4J_URI") or "bolt://localhost:7687",
        user=os.getenv("NEO4J_USER") or "neo4j",
        password=os.getenv("NEO4J_PASSWORD") or "password123",
        database=os.getenv("NEO4J_DATABASE") or "neo4j",
    )
    return DocumentIngestionService(
        DocumentRepository.from_store(store),
        ChunkRepository.from_store(store),
        vector_index,
        graph_index,
        ConversationRepository.from_store(store),
    )
