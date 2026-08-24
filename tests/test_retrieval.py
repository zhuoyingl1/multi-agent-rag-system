from multi_agent_rag.models import Document, RetrievalType
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.retrieval.store import LocalHybridStore


def build_retriever() -> HybridRetriever:
    documents = [
        Document(
            title="rag.md",
            text="RAG reduces hallucination by grounding answers in retrieved source evidence.",
        ),
        Document(
            title="metrics.md",
            text="Source coverage and grounding score are evaluation metrics for answer quality.",
        ),
        Document(
            title="graph.md",
            text="Neo4j can expand entities across related document chunks.",
        ),
    ]
    retriever = HybridRetriever()
    for document in documents:
        retriever.index(chunk_document(document))
    return retriever


def test_keyword_retrieval_returns_relevant_chunk() -> None:
    retriever = build_retriever()

    results = retriever.retrieve("grounding source evidence", top_k=2)

    assert results
    assert "grounding" in results[0].chunk.text.lower() or "source" in results[0].chunk.text.lower()
    assert results[0].retrieval_type in {RetrievalType.KEYWORD, RetrievalType.VECTOR}


def test_entity_retrieval_finds_named_backend() -> None:
    store = LocalHybridStore()
    document = Document(title="graph.md", text="Neo4j supports graph expansion for entity relationships.")
    store.add_chunks(chunk_document(document))

    results = store.entity_search("How does Neo4j help retrieval?", top_k=3)

    assert len(results) == 1
    assert results[0].retrieval_type is RetrievalType.ENTITY
    assert "neo4j" in results[0].highlights


def test_hybrid_retriever_deduplicates_chunks() -> None:
    retriever = build_retriever()

    results = retriever.retrieve("source grounding score", top_k=5)
    chunk_ids = [result.chunk.chunk_id for result in results]

    assert len(chunk_ids) == len(set(chunk_ids))
