# Demo Guide

This guide summarizes the runnable demo paths and expected outputs for the Multi-Agent RAG System V2 project.

## What This Project Demonstrates

- Structured document ingestion for Markdown, text, JSON, CSV, and extractable PDF files
- Shared text normalization for noisy extracted content, Unicode artifacts, ligatures, and PDF line wrapping
- Markdown-aware chunking for prose, code, formulas, and tables
- Local hybrid retrieval with keyword, vector-like, and entity expansion signals
- External Qdrant retrieval for production-style vector index persistence
- Neo4j graph indexing for chunk and entity relationships
- Deterministic planner, coordinator, specialist, grounding judge, and summarizer agents
- LangGraph adapter for production-style state graph orchestration
- Clean answer-only final responses with evidence, sources, and metrics exposed separately
- Evidence sufficiency fallback for unsupported or low-evidence queries
- Local Ollama answer composition for API and frontend natural-language answers
- Live Ollama service and model readiness checks for web demos
- FastAPI query, streaming, health, metrics, integration readiness, and evaluation endpoints
- Redis/Celery document ingestion with a local background-task development mode
- Optional JWT API authentication backed by MongoDB user accounts
- Next.js console for document queries, streaming answers, sources, metrics, integrations, and evaluation
- Deterministic evaluation runner for local regression testing

## Setup

From the repository root:

```powershell
python -m pip install -e ".[dev]"
docker compose up -d mongodb qdrant neo4j redis
python -m pytest -q
```

Expected test result:

```text
All tests passed
```

The exact runtime can vary by machine.

## CLI Demo

Inspect the implementation plan:

```powershell
python -m multi_agent_rag plan
```

Inspect chunking for the sample document:

```powershell
python -m multi_agent_rag ingest examples/sample_docs.md
```

Ask a grounded question:

```powershell
python -m multi_agent_rag ask "How does RAG reduce hallucination?"
```

The default API and frontend path uses LangGraph when the package is available. Force LangGraph from the CLI:

```powershell
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --orchestrator langgraph
```

The local deterministic orchestrator remains available as a test baseline:

```powershell
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --orchestrator local
```

The default retrieval path uses the external Qdrant service from `docker-compose.yml`. Neo4j is available as the graph relationship service. Start both before API, CLI, or frontend queries:

```powershell
docker compose up -d qdrant neo4j
```

Override the Qdrant connection details when using another service:

```powershell
$env:RETRIEVAL_BACKEND = "qdrant"
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_COLLECTION = "documents"
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --retrieval-backend qdrant
```

The local hybrid retriever remains available as a no-service test baseline:

```powershell
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --retrieval-backend local
```

Enable deterministic local reranking for a lightweight inspection path:

```powershell
$env:RERANKER_MODEL = "local"
$env:RERANKER_CANDIDATE_MULTIPLIER = "3"
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --retrieval-backend local
Remove-Item Env:RERANKER_MODEL
Remove-Item Env:RERANKER_CANDIDATE_MULTIPLIER
```

Enable BGE-style reranking after installing production dependencies:

```powershell
python -m pip install -e ".[production]"
$env:RERANKER_MODEL = "BAAI/bge-reranker-base"
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --retrieval-backend qdrant
```

When reranking is enabled, metrics include `reranker` and `candidate_sources`, and returned source items use the `reranked` retrieval type.

Enable local Ollama answer composition:

```powershell
ollama pull qwen2.5:3b
$env:LLM_ANSWER_PROVIDER = "ollama"
$env:LLM_ANSWER_MODEL = "qwen2.5:3b"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --retrieval-backend local
Remove-Item Env:LLM_ANSWER_PROVIDER
Remove-Item Env:LLM_ANSWER_MODEL
Remove-Item Env:OLLAMA_BASE_URL
```

When Ollama answer composition is enabled, metrics include `answer_type=llm` and `answer_model=qwen2.5:3b`. If the provider fails and `LLM_ANSWER_REQUIRED=false`, the workflow uses the deterministic answer fallback and records `answer_type=deterministic_fallback`.

API and frontend query requests set `require_llm_answer=true` by default and use local Ollama with `qwen2.5:3b` unless another provider configuration is supplied. If Ollama is unavailable, the API returns an explicit error instead of showing a deterministic fallback answer in the answer panel.

Index a document into Neo4j and inspect graph-expanded entities:

```powershell
python -m multi_agent_rag graph examples/sample_docs.md --query "How does Neo4j support RAG?"
```

Expected graph output shape:

```text
Document: sample_docs.md
Chunks indexed: 8
Related entities:
- ...
```

Expected answer shape:

```text
Question: How does RAG reduce hallucination?

Answer:
The strongest retrieved match is sample_docs.md...

Evidence:
- sample_docs.md: ...

Grounding score: 1.0
Unsupported claims: None
Sources: sample_docs.md
```

API responses expose workflow trace metadata for debugging and verification. The frontend keeps the query experience simple and uses the default orchestration path.

Ask an unsupported question:

```powershell
python -m multi_agent_rag ask "Who won the 1998 world chess championship?" --document examples/sample_docs.md
```

Expected fallback:

```text
No sufficiently relevant retrieved evidence was available, so the workflow cannot provide a grounded direct answer.
Grounding score: 0.0
Sources: no retrieved sources
```

## Evaluation Demo

Run the deterministic evaluation suite:

```powershell
python -m multi_agent_rag eval --retrieval-backend local --orchestrator local --min-pass-rate 1.0 --min-average-grounding 0.8 --output output/eval-report.json
```

Expected summary:

```text
cases: 7
passed: 7
failed: 0
pass_rate: 1.0
average_grounding_score: 1.0
total_failed_agents: 0
Quality gate: PASS
```

The JSON report contains per-case answers, missing expected terms, missing source terms, grounding score, retrieved source count, latency, failed-agent count, and configured quality-gate checks.

## Integration Readiness Demo

Check optional production integration readiness:

```powershell
python -m multi_agent_rag integrations
```

Expected local output:

```text
mode: local_with_optional_integrations
ready: 5/6
local_hybrid_store: ready
qdrant: ready
neo4j: ready
bge_reranker: missing_config
llm_answer: ready
langgraph: ready
```

The exact ready count depends on installed packages, environment variables, and running services. Qdrant and Neo4j are configured by default through `docker-compose.yml`; BGE reranking becomes ready after `RERANKER_MODEL` is set and `sentence-transformers` is installed. API integration readiness probes the local Ollama `/api/tags` endpoint and reports `llm_answer: ready` only when the configured answer model is installed.

## API Demo

Start the API:

```powershell
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

Open:

```text
http://127.0.0.1:8000/docs
```

Open Neo4j Browser:

```text
http://localhost:7474
```

Default local credentials:

```text
username: neo4j
password: password123
```

Useful endpoints:

- `GET /health`
- `GET /health/metrics`
- `GET /health/integrations`
- `GET /health/liveness`
- `GET /health/readiness`
- `GET /health/task-queue`
- `POST /auth/register`
- `POST /auth/token`
- `GET /auth/me`
- `POST /documents/upload`
- `POST /query`
- `POST /query/stream`
- `POST /evaluate`

Use the local document task backend when running only the API:

```powershell
$env:DOCUMENT_TASK_BACKEND = "local"
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

For durable document ingestion, start Redis and a worker before the API:

```powershell
docker compose up -d redis
$env:DOCUMENT_TASK_BACKEND = "celery"
python -m celery -A multi_agent_rag.celery_app:celery_app worker --loglevel=INFO --pool=solo
```

Sample `/query` request:

```json
{
  "query": "How does RAG reduce hallucination?",
  "document_path": "examples/sample_docs.md",
  "orchestrator": "langgraph",
  "retrieval_backend": "qdrant"
}
```

Sample `/evaluate` request:

```json
{}
```

Expected `/evaluate` result:

```json
{
  "case_count": 7,
  "passed_count": 7,
  "failed_count": 0,
  "pass_rate": 1.0
}
```

Sample upload flow:

```powershell
curl -X POST -F "file=@examples/sample_docs.md" http://127.0.0.1:8000/documents/upload
```

The response includes a `document_path` value that can be reused in `/query` and `/query/stream`.

## Frontend Demo

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:3000
```

Demo checks:

- Upload a supported document and confirm the `Document path` field updates.
- Run `Query` mode and inspect the final grounded answer.
- Run `Stream` mode and watch `answer_delta` events render progressively.
- Inspect source snippets and retrieval highlights.
- Check `Integrations` for local and optional production readiness.
- Click `Run Eval` in the `Evaluation` panel and confirm `3/3 passed`.

## Current Limitations

- Local vector retrieval is vector-like lexical scoring, not a learned embedding model.
- Qdrant and Neo4j require Docker services for the production-style path.
- BGE reranking requires `sentence-transformers` and model download access when using a real model name.
- Ollama answer composition requires a running local Ollama service and an available model.
- PDF ingestion depends on extractable text and does not perform OCR.
- The deterministic judge uses lexical overlap and should be replaced or augmented with an LLM judge for production evaluation.
