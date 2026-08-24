# Multi-Agent RAG System V2

A clean, from-scratch multi-agent RAG prototype built as a step-by-step learning and portfolio project.

## Current Status

Step 1 adds core document models and Markdown-aware structured chunking. The project can now convert a source document into stable prose, code, table, and formula chunks.

## Implemented Capabilities

- Python package scaffold with CLI entry point
- Local test harness with pytest
- Core `Document` and `Chunk` models
- Stable document and chunk identifiers
- Structured chunking for prose, fenced code blocks, Markdown tables, and formula blocks

## Planned Capabilities

- Document ingestion for local demo documents and uploaded files
- Local hybrid retrieval with keyword, vector-like, and entity expansion signals
- Planner, coordinator, expert, grounding judge, and summarizer agents
- FastAPI endpoints with Server-Sent Events for inspectable workflow progress
- Evaluation metrics, health checks, and deterministic fallback behavior
- Optional production-style integrations for LangGraph, Qdrant, Neo4j, and reranking

## Quick Start

```powershell
$env:PYTHONPATH = "D:\my_projects\multi-agent-rag-system-v2\src"
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
