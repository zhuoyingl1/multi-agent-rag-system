# Multi-Agent RAG System V2

[![CI](https://github.com/zhuoyingl1/multi-agent-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/zhuoyingl1/multi-agent-rag-system/actions/workflows/ci.yml)

A clean, from-scratch multi-agent RAG prototype built as a step-by-step learning and portfolio project.

## Current Status

The application now supports persistent, versioned document indexing across MongoDB, Qdrant, and Neo4j, durable Redis/Celery ingestion jobs, grounded multi-turn answers through Ollama, validated runtime retrieval settings, full-stack containers, and automated CI checks.

## Implemented Capabilities

- Python package scaffold with CLI entry point
- Local test harness with pytest
- Core `Document` and `Chunk` models
- Stable document and chunk identifiers
- Structured chunking for prose, fenced code blocks, Markdown tables, and formula blocks
- Bounded table chunking with repeated headers and source row ranges
- Local hybrid retrieval with keyword, vector-like, and entity expansion signals
- Retrieval result scoring, highlights, and chunk-level deduplication
- Deterministic planner, coordinator, expert, grounding judge, and summarizer agents
- Local `ask` command for an end-to-end workflow demo
- Document ingestion for text, Markdown, JSON, CSV, DOCX, PPTX, XLSX, HTML, and basic text-based PDF files
- DOCX structure preservation for headings, lists, and tables
- PPTX structure preservation for slide titles, text, tables, and slide-level citations
- XLSX structure preservation for worksheets, tabular rows, formulas, and section-aware citations
- HTML structure preservation for headings, prose, lists, code blocks, tables, and visible-content filtering
- Leaf-level JSON chunking with exact object and array path citations
- PDF text cleanup for common extraction artifacts such as private-use bullets and replacement characters
- Shared text normalization for Unicode artifacts, ligatures, private-use characters, control characters, and noisy PDF line wrapping
- Local `ingest` command for document loading and chunking inspection
- UTF-8 CLI output configuration for PDF text with special characters
- Clean answer-only final responses with evidence and metrics exposed separately
- Direct answer formatting for project-style and evidence-backed questions
- FastAPI application with health, document, conversation, query, streaming, and evaluation endpoints
- In-memory metrics registry for local observability
- Deployment liveness and concurrent dependency readiness probes
- Configurable local or Redis/Celery document task dispatch with live worker readiness
- Optional JWT API authentication with scrypt password hashing and MongoDB user accounts
- Per-user ownership isolation for knowledge spaces, documents, conversations, and registered-document queries
- Frontend registration, login, tab-scoped token persistence, automatic Bearer requests, and sign-out
- Next.js console for query, streaming, metrics, and source inspection
- SSE workflow events with progressive answer delta rendering
- Local evaluation runner with JSON cases and JSON report export
- Rank-based retrieval evaluation with Recall@K, Precision@K, MRR, nDCG@K, and latency metrics
- Evidence sufficiency gate with grounded fallback responses
- Production integration readiness checks for LangGraph, Qdrant, Neo4j, and reranking
- FastAPI and frontend integration readiness dashboard
- FastAPI and frontend evaluation dashboard
- Consistent API error handling for document loading and evaluation failures
- Request validation for empty query, document, and evaluation inputs
- Document upload API and frontend upload workflow
- LangGraph workflow adapter with CLI and API orchestration support
- Internal workflow trace metadata for orchestration mode, evidence status, agents, and retrieved sources
- Qdrant retrieval adapter and retrieval backend factory for production vector search
- Neo4j graph adapter for document chunk and entity relationship indexing
- Configurable reranking wrapper for local, Qdrant, and BGE-style candidate reranking
- Local Ollama answer composer with API/frontend LLM answer enforcement and answer provider metrics
- Live Ollama service and model readiness reporting in the integration dashboard
- Persistent document metadata and structured chunks in MongoDB
- Document-scoped vectors in Qdrant and entity relationships in Neo4j
- Index version metadata with stale-index detection and explicit reindexing
- Idempotent replacement of MongoDB chunks, Qdrant vectors, and Neo4j graph nodes
- Paginated document catalog with status filtering
- Persistent document display names with inline rename controls
- Cross-store document deletion for files, chunks, vectors, graph nodes, and conversations
- Searchable, type-filtered indexed chunk previews with source locations
- MongoDB-backed knowledge spaces with multi-document Qdrant, keyword, and Neo4j retrieval
- SSE document progress updates with polling fallback and failed-index retry
- Sequential batch upload queue with per-document progress and failure isolation
- Persistent document-scoped conversations with context-aware follow-up queries
- Scoped conversation history with question-based titles, message review, rename, and deletion
- Token-level Ollama answer streaming over SSE
- Source citations with page, line, row, section, and JSON-path locators
- GitHub-flavored Markdown answers with clickable source citations
- Full-stack Docker Compose deployment with service health checks and shared worker storage
- GitHub Actions checks for backend tests, frontend builds, and container configuration
- Runtime settings API with MongoDB persistence, TTL caching, version checks, and request-stable snapshots

## Planned Capabilities

- Larger benchmark coverage and managed deployment templates

## Quick Start

```powershell
$env:PYTHONPATH = "D:\my_projects\multi-agent-rag-system-v2\src"
python -m pip install -e ".[dev]"
python -m pytest -q
python -m multi_agent_rag --help
python -m multi_agent_rag plan
python -m multi_agent_rag ingest examples/sample_docs.md
python -m multi_agent_rag ingest examples/sample_docs.md --show-chunks
python -m multi_agent_rag ask "How does RAG reduce hallucination?"
python -m multi_agent_rag eval --output output/eval-report.json
python -m multi_agent_rag retrieval-eval --retrieval-backend local --output output/retrieval-eval-report.json
python -m multi_agent_rag integrations
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

Evaluation commands can enforce minimum quality levels and return a non-zero process exit code when a threshold fails. The checked-in CI benchmark uses deterministic local retrieval so it does not depend on external services or paid APIs:

```powershell
python -m multi_agent_rag eval --orchestrator local --retrieval-backend local --min-pass-rate 1.0 --min-average-grounding 0.8
python -m multi_agent_rag retrieval-eval --retrieval-backend local --top-k 5 --min-pass-rate 1.0 --min-average-recall 1.0 --min-mrr 0.6 --min-ndcg 0.7
```

Both commands accept `--max-average-latency-ms` for controlled performance environments. Latency is reported but not enforced in shared CI because runner performance is variable. JSON reports include every configured quality-gate check, its threshold, actual value, and pass status.

Runtime retrieval settings can be inspected with `GET /settings/runtime` and partially updated with `PUT /settings/runtime`. The API accepts only documented non-sensitive settings such as `top_k`, query rewriting, adaptive result selection, context budget, reranker candidate count, and RRF strength. Updates are atomic and affect new requests only. Secrets, service URLs, model names, and credentials remain environment-only configuration. Local runs default to the in-memory backend. Docker Compose enables the MongoDB backend, which persists settings across restarts, checks revisions to prevent stale writes, and refreshes each API process through a configurable TTL cache.

Deployment probes are available at `GET /health/liveness` for process health and
`GET /health/readiness` for live MongoDB, Qdrant, Neo4j, Ollama, and task queue readiness.
`GET /health/task-queue` reports the selected task backend and responsive worker count.

The default document task backend is local so the application remains easy to run. For durable jobs, start Redis and a Celery worker, then use the same task backend setting for the API:

```powershell
docker compose up -d redis
$env:DOCUMENT_TASK_BACKEND = "celery"
python -m celery -A multi_agent_rag.celery_app:celery_app worker --loglevel=INFO --pool=solo
```

In a separate terminal:

```powershell
$env:DOCUMENT_TASK_BACKEND = "celery"
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

Enable API authentication with a private secret of at least 32 characters:

```powershell
$env:AUTH_REQUIRED = "true"
$env:JWT_SECRET = "replace-with-a-long-random-production-secret"
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

Register with `POST /auth/register`, obtain a new token with `POST /auth/token`, and verify it with `GET /auth/me`. In the API documentation, use the `Authorize` control and enter the access token before calling protected endpoints. Set `AUTH_ALLOW_REGISTRATION=false` after provisioning users when public registration is not desired.

When authentication is enabled, knowledge spaces, uploaded documents, and conversations are owned by the authenticated user. Resource lookups are filtered by owner and inaccessible records return `404`. Authenticated queries must use a registered `document_id` or `knowledge_space_id`; arbitrary server-local document paths and local-path evaluation are disabled. Records created before ownership support are unowned and remain available only while `AUTH_REQUIRED=false`; re-upload them under the intended account before enabling authentication.

Enable local LLM answer composition with Ollama for CLI demos:

```powershell
$env:LLM_ANSWER_PROVIDER = "ollama"
$env:LLM_ANSWER_MODEL = "qwen2.5:3b"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --retrieval-backend local
```

API and frontend query requests require Ollama answers by default. Start Ollama and pull the default model before using the web console:

```powershell
ollama pull qwen2.5:3b
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

Run the frontend in a second terminal:

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_AUTH_REQUIRED = "true"
npm run dev
```

The frontend stores the access token in `sessionStorage`, restores valid sessions after a page refresh, and clears invalid tokens after a `401` response. Set `NEXT_PUBLIC_AUTH_REQUIRED=false` to keep the unauthenticated local-development workflow available.

## Full Stack Containers

The Compose stack runs the API, Celery worker, frontend, MongoDB, Qdrant, Neo4j, and Redis. It keeps uploaded documents and Hugging Face model files in shared named volumes so the API and worker see the same files and model downloads survive container restarts.

Ollama remains on the host to avoid duplicating local model storage. Pull the answer and embedding models before starting the stack:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

If Ollama is not reachable from Docker Desktop, allow it to listen beyond host loopback and restart Ollama:

```powershell
$env:OLLAMA_HOST = "0.0.0.0:11434"
ollama serve
```

Create the local environment file, review its secrets, and start the stack:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health/liveness
Invoke-RestMethod http://127.0.0.1:8000/health/readiness
```

Open the web console at `http://127.0.0.1:3000` and the API documentation at `http://127.0.0.1:8000/docs`. Follow service output or stop the stack with:

```powershell
docker compose logs -f api worker
docker compose down
```

Set `INSTALL_PRODUCTION_EXTRAS=false` and `RERANKER_MODEL=lexical` for a smaller development image without the BGE reranker. Keep `INSTALL_PRODUCTION_EXTRAS=true` for the production-like BGE configuration. If authentication is enabled, set `AUTH_REQUIRED=true`, set a strong `JWT_SECRET`, and keep the frontend build setting aligned by rebuilding the frontend service.

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







