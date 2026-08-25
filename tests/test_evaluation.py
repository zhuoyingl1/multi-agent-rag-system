import json

from multi_agent_rag.evaluation import load_eval_cases, run_evaluation


def test_load_eval_cases_reads_json(tmp_path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case_one",
                        "query": "What is RAG?",
                        "expected_terms": ["RAG"],
                        "required_source_terms": ["evidence"],
                        "min_grounding_score": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(cases_path)

    assert len(cases) == 1
    assert cases[0].case_id == "case_one"
    assert cases[0].expected_terms == ["RAG"]


def test_run_evaluation_reports_pass_rate(tmp_path) -> None:
    document_path = tmp_path / "docs.md"
    document_path.write_text("RAG reduces hallucination by grounding answers in source evidence.", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "rag_grounding",
                        "query": "How does RAG reduce hallucination?",
                        "expected_terms": ["RAG", "source evidence"],
                        "required_source_terms": ["source evidence"],
                        "min_grounding_score": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_evaluation(document_path, cases_path)

    assert report.case_count == 1
    assert report.passed_count == 1
    assert report.pass_rate == 1.0
    assert report.average_retrieved_sources >= 1
