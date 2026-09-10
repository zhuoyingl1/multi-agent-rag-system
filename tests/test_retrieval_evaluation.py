import json

import pytest

from multi_agent_rag.models import Chunk, ChunkType, RetrievalType, SearchResult
from multi_agent_rag.retrieval_evaluation import load_retrieval_eval_cases, run_retrieval_evaluation


def test_load_retrieval_eval_cases_validates_and_deduplicates_indices(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "grounding",
                        "query": "How is the answer grounded?",
                        "relevant_chunk_indices": [2, 1, 2],
                        "min_recall_at_k": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_retrieval_eval_cases(path)

    assert cases[0].relevant_chunk_indices == [1, 2]
    assert cases[0].min_recall_at_k == 0.5


def test_load_retrieval_eval_cases_rejects_missing_gold_chunks(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"case_id": "invalid", "query": "Question"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="relevant_chunk_indices"):
        load_retrieval_eval_cases(path)


def test_run_retrieval_evaluation_reports_rank_metrics(tmp_path) -> None:
    document_path = tmp_path / "notes.md"
    document_path.write_text(
        "# Notes\n\nRAG grounds answers in retrieved evidence.\n\n# Operations\n\nLatency measures response speed.",
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "rag_evidence",
                        "query": "How does RAG use retrieved evidence?",
                        "relevant_chunk_indices": [1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_retrieval_evaluation(document_path, cases_path, retrieval_backend="local", top_k=3)

    assert report.retrieval_backend == "local"
    assert report.case_count == 1
    assert report.passed_count == 1
    assert report.average_recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.cases[0].relevant_ranks == [1]
    assert report.cases[0].retrieved_chunk_indices[0] == 1


def test_rank_metrics_penalize_relevant_chunks_that_appear_later(monkeypatch, tmp_path) -> None:
    class StubRetriever:
        document_id = ""

        def index(self, chunks) -> None:
            self.document_id = chunks[0].document_id

        def retrieve(self, _query, top_k=None):
            chunks = [
                Chunk("foreign-doc", "foreign-chunk", "Foreign chunk", ChunkType.PROSE, 1),
                Chunk(self.document_id, "chunk-2", "Chunk 2", ChunkType.PROSE, 2),
                Chunk(self.document_id, "chunk-1", "Chunk 1", ChunkType.PROSE, 1),
            ]
            return [SearchResult(chunk, 1.0, RetrievalType.HYBRID) for chunk in chunks[:top_k]]

    monkeypatch.setattr(
        "multi_agent_rag.retrieval_evaluation.create_retriever",
        lambda _backend, top_k: StubRetriever(),
    )
    document_path = tmp_path / "notes.md"
    document_path.write_text("# Notes\n\nEvidence.\n\n# Details", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "late_evidence",
                        "query": "Find evidence",
                        "relevant_chunk_indices": [1, 2],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_retrieval_evaluation(document_path, cases_path, retrieval_backend="local", top_k=3).cases[0]

    assert result.relevant_ranks == [2, 3]
    assert result.retrieved_document_ids[0] == "foreign-doc"
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 0.6667
    assert result.reciprocal_rank == 0.5
    assert result.ndcg_at_k == 0.6934


def test_run_retrieval_evaluation_rejects_invalid_top_k(tmp_path) -> None:
    with pytest.raises(ValueError, match="top_k"):
        run_retrieval_evaluation(tmp_path / "document.md", tmp_path / "cases.json", top_k=0)


def test_run_retrieval_evaluation_rejects_unavailable_gold_index(tmp_path) -> None:
    document_path = tmp_path / "notes.md"
    document_path.write_text("Only one chunk.", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "stale_gold",
                        "query": "Find evidence",
                        "relevant_chunk_indices": [99],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unavailable chunk indices: 99"):
        run_retrieval_evaluation(document_path, cases_path, retrieval_backend="local")
