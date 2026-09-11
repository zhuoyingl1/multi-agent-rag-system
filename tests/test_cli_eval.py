import json
import subprocess
import sys

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
    assert "Quality gate: not configured" in output
    assert report["passed_count"] == 1
    assert report["quality_gate"] == {"passed": True, "checks": []}


def test_cli_eval_returns_failure_when_quality_gate_misses(tmp_path, capsys) -> None:
    document_path = tmp_path / "docs.md"
    document_path.write_text("RAG grounds answers in source evidence.", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "missing_answer_term",
                        "query": "How is RAG grounded?",
                        "expected_terms": ["term that cannot be present"],
                        "required_source_terms": ["source evidence"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "failed-report.json"

    exit_code = main(
        [
            "eval",
            "--document",
            str(document_path),
            "--cases",
            str(cases_path),
            "--min-pass-rate",
            "1.0",
            "--output",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert "Quality gate: FAIL" in output
    assert report["quality_gate"]["passed"] is False
    assert report["quality_gate"]["checks"][0]["metric"] == "pass_rate"


def test_module_entrypoint_propagates_quality_gate_failure(tmp_path) -> None:
    document_path = tmp_path / "docs.md"
    document_path.write_text("RAG grounds answers in source evidence.", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "failed_gate",
                        "query": "How is RAG grounded?",
                        "expected_terms": ["missing expected value"],
                        "required_source_terms": ["source evidence"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "multi_agent_rag",
            "eval",
            "--document",
            str(document_path),
            "--cases",
            str(cases_path),
            "--orchestrator",
            "local",
            "--retrieval-backend",
            "local",
            "--min-pass-rate",
            "1.0",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "Quality gate: FAIL" in completed.stdout


def test_cli_retrieval_eval_writes_passing_quality_gate(tmp_path, capsys) -> None:
    document_path = tmp_path / "docs.md"
    document_path.write_text("# Notes\n\nRAG grounds answers in retrieved evidence.", encoding="utf-8")
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
    output_path = tmp_path / "retrieval-report.json"

    exit_code = main(
        [
            "retrieval-eval",
            "--document",
            str(document_path),
            "--cases",
            str(cases_path),
            "--retrieval-backend",
            "local",
            "--top-k",
            "3",
            "--min-pass-rate",
            "1.0",
            "--min-average-recall",
            "1.0",
            "--output",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Quality gate: PASS" in output
    assert report["quality_gate"]["passed"] is True
    assert len(report["quality_gate"]["checks"]) == 2
