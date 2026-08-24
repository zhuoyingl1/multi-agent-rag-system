"""Command line interface for the Multi-Agent RAG System V2."""

from __future__ import annotations

import argparse

from multi_agent_rag import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi_agent_rag",
        description="Multi-Agent RAG System V2 command line tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Show the planned implementation stages.")
    plan_parser.set_defaults(func=print_plan)

    return parser


def print_plan(_args: argparse.Namespace) -> int:
    stages = [
        "1. Project scaffold and local test harness",
        "2. Document models and structured chunking",
        "3. Local hybrid retrieval",
        "4. Multi-agent workflow",
        "5. API, streaming, metrics, and demo evidence",
    ]
    print("Implementation plan:")
    for stage in stages:
        print(f"- {stage}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)
