# Multi-Agent RAG System V2

A clean, from-scratch multi-agent RAG prototype built as a step-by-step learning and portfolio project.

## Current Status

Step 19 adds an optional LangGraph orchestration adapter while keeping the deterministic local workflow as the default runtime.

## Implemented Capabilities

- Python package scaffold with CLI entry point
- Local test harness with pytest
- Core `Document` and `Chunk` models
- Stable document and chunk identifiers
- Structured chunking for prose, fenced code blocks, Markdown tables, and formula blocks
- Local hybrid retrieval with keyword, vector-like, and entity expansion signals
- Retrieval result scoring, highlights, and chunk-level deduplication
- Deterministic planner, coordinator, expert, grounding judge, and summarizer agents
- Local `ask` command for an end-to-end workflow demo
- Document ingestion for text, Markdown, JSON, CSV, and basic text-based PDF files
- PDF text cleanup for common extraction artifacts such as private-use bullets and replacement characters
- Local `ingest` command for document loading and chunking inspection
- UTF-8 CLI output configuration for PDF text with special characters
- Structured deterministic answers with evidence and unsupported-claim reporting
- Direct answer formatting for project-style and evidence-backed questions
- FastAPI application with `/health`, `/health/metrics`, `/query`, and `/query/stream`
- In-memory metrics registry for local observability
- Next.js console for query, streaming, metrics, and source inspection
- SSE workflow events with progressive answer delta rendering
- Local evaluation runner with JSON cases and JSON report export
- Evidence sufficiency gate with grounded fallback responses
- Optional production integration readiness checks for LangGraph, Qdrant, Neo4j, and reranking
- FastAPI and frontend integration readiness dashboard
- FastAPI and frontend evaluation dashboard
- Consistent API error handling for document loading and evaluation failures
- Request validation for empty query, document, and evaluation inputs
- Document upload API and frontend upload workflow
- Optional LangGraph workflow adapter with CLI, API, and frontend orchestration selection

## Planned Capabilities

- Production adapter implementations backed by live external services

## Quick Start

```powershell
$env:PYTHONPATH = "D:\my_projects\multi-agent-rag-system-v2\src"
python -m pip install -e ".[dev]"
python -m pytest -q
python -m multi_agent_rag --help
python -m multi_agent_rag plan
python -m multi_agent_rag ingest examples/sample_docs.md
python -m multi_agent_rag ask "How does RAG reduce hallucination?"
python -m multi_agent_rag eval --output output/eval-report.json
python -m multi_agent_rag integrations
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

Run the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

## Demo Guide

See [docs/demo-guide.md](docs/demo-guide.md) for CLI, API, frontend, evaluation, and integration readiness demo notes.

## Project Layout

```text
src/multi_agent_rag/   Python package
examples/              Demo documents and sample inputs
tests/                 Local tests that avoid external services
frontend/              Next.js local console
docs/                  Demo and project notes
```

## Development Approach

This project is intentionally built in small commits. Each step should add one clear capability, include focused tests, and keep the project runnable without paid APIs or external databases.







