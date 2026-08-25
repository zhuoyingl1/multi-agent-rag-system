import json

from multi_agent_rag.cli import main


def test_cli_eval_writes_report(tmp_path, capsys) -> None:
    document_path = tmp_path / "docs.md"
    document_path.write_text("Source coverage and latency can monitor RAG answer quality.", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "quality_metrics",
                        "query": "What metrics monitor answer quality?",
                        "expected_terms": ["Source coverage", "latency"],
                        "required_source_terms": ["Source coverage", "latency"],
                        "min_grounding_score": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    exit_code = main(["eval", "--document", str(document_path), "--cases", str(cases_path), "--output", str(output_path)])

    output = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Evaluation report:" in output
    assert "quality_metrics: PASS" in output
    assert report["passed_count"] == 1
