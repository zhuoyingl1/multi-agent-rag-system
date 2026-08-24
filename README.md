# Multi-Agent RAG System V2

A clean, from-scratch multi-agent RAG prototype built as a step-by-step learning and portfolio project.

## Current Status

Step 0 creates the project scaffold only. It verifies that the package can be imported, the CLI can run, and the repository is ready for incremental implementation.

## Planned Capabilities

- Document ingestion for local demo documents and uploaded files
- Structured chunking for prose, code, formulas, and tables
- Local hybrid retrieval with keyword, vector-like, and entity expansion signals
- Planner, coordinator, expert, grounding judge, and summarizer agents
- FastAPI endpoints with Server-Sent Events for inspectable workflow progress
- Evaluation metrics, health checks, and deterministic fallback behavior
- Optional production-style integrations for LangGraph, Qdrant, Neo4j, and reranking

## Quick Start

```powershell
python -m pytest -q
python -m multi_agent_rag --help
python -m multi_agent_rag plan
```

## Project Layout

```text
src/multi_agent_rag/   Python package
examples/              Demo documents and sample inputs
tests/                 Local tests that avoid external services
```

## Development Approach

This project is intentionally built in small commits. Each step should add one clear capability, include focused tests, and keep the project runnable without paid APIs or external databases.
