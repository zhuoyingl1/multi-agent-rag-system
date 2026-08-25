# Demo Guide

This guide summarizes the runnable demo paths and expected outputs for the Multi-Agent RAG System V2 project.

## What This Project Demonstrates

- Structured document ingestion for Markdown, text, JSON, CSV, and extractable PDF files
- Markdown-aware chunking for prose, code, formulas, and tables
- Local hybrid retrieval with keyword, vector-like, and entity expansion signals
- External Qdrant retrieval for production-style vector index persistence
- Neo4j graph indexing for chunk and entity relationships
- Deterministic planner, coordinator, specialist, grounding judge, and summarizer agents
- LangGraph adapter for production-style state graph orchestration
- Grounded answer formatting with evidence, sources, and unsupported-claim reporting
- Evidence sufficiency fallback for unsupported or low-evidence queries
- FastAPI query, streaming, health, metrics, integration readiness, and evaluation endpoints
- Next.js console for document queries, streaming answers, sources, metrics, integrations, and evaluation
- Deterministic evaluation runner for local regression testing

## Setup

From the repository root:

```powershell
python -m pip install -e ".[dev]"
docker compose up -d qdrant neo4j
python -m pytest -q
```

Expected test result:

```text
62 passed
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

Index a document into Neo4j and inspect graph-expanded entities:

```powershell
python -m multi_agent_rag graph examples/sample_docs.md --query "How does Neo4j support RAG?"
```

Expected graph output shape:

```text
Document: sample_docs.md
Chunks indexed: 5
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
python -m multi_agent_rag eval --output output/eval-report.json
```

Expected summary:

```text
cases: 3
passed: 3
failed: 0
pass_rate: 1.0
average_grounding_score: 1.0
total_failed_agents: 0
```

The JSON report contains per-case answers, missing expected terms, missing source terms, grounding score, retrieved source count, latency, and failed-agent count.

## Integration Readiness Demo

Check optional production integration readiness:

```powershell
python -m multi_agent_rag integrations
```

Expected local output:

```text
mode: local_with_optional_integrations
ready: 4/5
local_hybrid_store: ready
qdrant: ready
neo4j: ready
bge_reranker: missing_config
langgraph: ready
```

The exact ready count depends on installed packages and environment variables. Qdrant and Neo4j are configured by default through `docker-compose.yml`; reranking remains an optional production integration.

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
- `POST /documents/upload`
- `POST /query`
- `POST /query/stream`
- `POST /evaluate`

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
  "case_count": 3,
  "passed_count": 3,
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
- The reranker production adapter remains a readiness boundary rather than a live service implementation.
- PDF ingestion depends on extractable text and does not perform OCR.
- The deterministic judge uses lexical overlap and should be replaced or augmented with an LLM judge for production evaluation.
