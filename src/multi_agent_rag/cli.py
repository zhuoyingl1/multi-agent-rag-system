"""Command line interface for the Multi-Agent RAG System V2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from multi_agent_rag import __version__
from multi_agent_rag.documents import load_document
from multi_agent_rag.evaluation import run_evaluation
from multi_agent_rag.integrations import check_integrations
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow


def configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi_agent_rag",
        description="Multi-Agent RAG System V2 command line tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Show the planned implementation stages.")
    plan_parser.set_defaults(func=print_plan)

    ask_parser = subparsers.add_parser("ask", help="Run a local deterministic multi-agent RAG demo.")
    ask_parser.add_argument("query", help="Research question to answer from the demo documents.")
    ask_parser.add_argument("--document", default="examples/sample_docs.md", help="Path to a local supported document.")
    ask_parser.set_defaults(func=run_ask)

    ingest_parser = subparsers.add_parser("ingest", help="Load and chunk a local supported document.")
    ingest_parser.add_argument("document", help="Path to a local supported document.")
    ingest_parser.set_defaults(func=run_ingest)

    eval_parser = subparsers.add_parser("eval", help="Run deterministic local evaluation cases.")
    eval_parser.add_argument("--document", default="examples/sample_docs.md", help="Path to a local supported document.")
    eval_parser.add_argument("--cases", default="examples/eval_cases.json", help="Path to evaluation cases JSON.")
    eval_parser.add_argument("--output", help="Optional path for the JSON evaluation report.")
    eval_parser.set_defaults(func=run_eval)

    integrations_parser = subparsers.add_parser("integrations", help="Show optional production integration readiness.")
    integrations_parser.add_argument("--json", action="store_true", help="Print the readiness report as JSON.")
    integrations_parser.set_defaults(func=run_integrations)

    return parser


def print_plan(_args: argparse.Namespace) -> int:
    stages = [
        "1. Project scaffold and local test harness",
        "2. Document models and structured chunking",
        "3. Local hybrid retrieval",
        "4. Multi-agent workflow and document ingestion",
        "5. API, streaming, metrics, and demo evidence",
        "6. Frontend console and deterministic evaluation runner",
        "7. Optional production integration readiness path",
    ]
    print("Implementation plan:")
    for stage in stages:
        print(f"- {stage}")
    return 0

def run_ask(args: argparse.Namespace) -> int:
    document = load_document(Path(args.document))
    retriever = HybridRetriever()
    retriever.index(chunk_document(document))
    result = MultiAgentRAGWorkflow(retriever).run(args.query)

    print(result.answer)
    print("\nAgents:")
    for agent in result.agents:
        print(f"- {agent.agent_name}: confidence={agent.confidence}")
    print("\nMetrics:")
    for key, value in result.metrics.items():
        print(f"- {key}: {value}")
    return 0

def run_ingest(args: argparse.Namespace) -> int:
    document = load_document(Path(args.document))
    chunks = chunk_document(document)
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.chunk_type.value] = counts.get(chunk.chunk_type.value, 0) + 1

    print(f"Document: {document.title}")
    print(f"Type: {document.metadata['document_type']}")
    print(f"Characters: {len(document.text)}")
    print(f"Chunks: {len(chunks)}")
    for chunk_type, count in sorted(counts.items()):
        print(f"- {chunk_type}: {count}")
    return 0


def run_eval(args: argparse.Namespace) -> int:
    report = run_evaluation(Path(args.document), Path(args.cases))
    print("Evaluation report:")
    print(f"- document: {report.document_path}")
    print(f"- cases: {report.case_count}")
    print(f"- passed: {report.passed_count}")
    print(f"- failed: {report.failed_count}")
    print(f"- pass_rate: {report.pass_rate}")
    print(f"- average_grounding_score: {report.average_grounding_score}")
    print(f"- average_latency_ms: {report.average_latency_ms}")
    print(f"- average_retrieved_sources: {report.average_retrieved_sources}")
    print(f"- total_failed_agents: {report.total_failed_agents}")
    print("Cases:")
    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        print(
            f"- {case.case_id}: {status} "
            f"grounding={case.grounding_score} sources={case.retrieved_sources} latency_ms={case.latency_ms}"
        )
        if case.missing_expected_terms:
            print(f"  missing_expected_terms: {', '.join(case.missing_expected_terms)}")
        if case.missing_source_terms:
            print(f"  missing_source_terms: {', '.join(case.missing_source_terms)}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"JSON report written to {output_path}")
    return 0


def run_integrations(args: argparse.Namespace) -> int:
    report = check_integrations()
    if args.json:
        print(report.to_json())
        return 0

    print("Integration readiness:")
    print(f"- mode: {report.mode}")
    print(f"- ready: {report.ready_count}/{report.integration_count}")
    for integration in report.integrations:
        package = integration.required_package or "built-in"
        print(f"- {integration.name}: {integration.status} package={package}")
        print(f"  role: {integration.role}")
        print(f"  notes: {integration.notes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)




