"""Command line interface for the Multi-Agent RAG System V2."""

from __future__ import annotations

import argparse
from pathlib import Path

from multi_agent_rag import __version__
from multi_agent_rag.documents import load_document
from multi_agent_rag.retrieval.chunking import chunk_document
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow


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

    return parser


def print_plan(_args: argparse.Namespace) -> int:
    stages = [
        "1. Project scaffold and local test harness",
        "2. Document models and structured chunking",
        "3. Local hybrid retrieval",
        "4. Multi-agent workflow and document ingestion",
        "5. API, streaming, metrics, and demo evidence",
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

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)



