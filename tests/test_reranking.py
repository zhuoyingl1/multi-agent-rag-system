from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.retrieval.reranking import LexicalReranker, RerankingRetriever


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.requested_top_k: int | None = None
        self.closed = False

    def index(self, chunks: list[Chunk]) -> None:
        return None

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.requested_top_k = top_k
        return self.results

    def close(self) -> None:
        self.closed = True


def result(text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            document_id="doc",
            chunk_id=f"chunk_{abs(hash(text))}",
            text=text,
            chunk_type=ChunkType.PROSE,
            index=0,
            metadata={"title": "demo.md"},
        ),
        score=score,
        retrieval_type=RetrievalType.KEYWORD,
        highlights=[],
    )


def test_lexical_reranker_prioritizes_query_term_coverage() -> None:
    reranker = LexicalReranker()
    weak = result("General retrieval notes.", score=0.95)
    strong = result("RAG reduces hallucination with source evidence.", score=0.1)

    reranked = reranker.rerank("How does RAG reduce hallucination?", [weak, strong], top_k=1)

    assert reranked == [reranked[0]]
    assert reranked[0].chunk.text == strong.chunk.text
    assert reranked[0].retrieval_type is RetrievalType.RERANKED
    assert "reranked" in reranked[0].highlights


def test_reranking_retriever_expands_candidate_pool_and_closes_base() -> None:
    base = FakeRetriever(
        [
            result("RAG grounding evidence.", score=0.5),
            result("Metrics and latency.", score=0.4),
        ]
    )
    retriever = RerankingRetriever(base, LexicalReranker(), candidate_multiplier=4)

    reranked = retriever.retrieve("RAG evidence", top_k=2)
    retriever.close()

    assert base.requested_top_k == 8
    assert retriever.last_candidate_count == 2
    assert retriever.last_reranker == "lexical"
    assert len(reranked) == 2
    assert base.closed is True
