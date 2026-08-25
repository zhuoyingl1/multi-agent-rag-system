# Demo Guide

This guide summarizes the runnable demo paths and expected outputs for the Multi-Agent RAG System V2 project.

## What This Project Demonstrates

- Structured document ingestion for Markdown, text, JSON, CSV, and extractable PDF files
- Markdown-aware chunking for prose, code, formulas, and tables
- Local hybrid retrieval with keyword, vector-like, and entity expansion signals
- Deterministic planner, coordinator, specialist, grounding judge, and summarizer agents
- Optional LangGraph adapter for production-style state graph orchestration
- Grounded answer formatting with evidence, sources, and unsupported-claim reporting
- Evidence sufficiency fallback for unsupported or low-evidence queries
- FastAPI query, streaming, health, metrics, integration readiness, and evaluation endpoints
- Next.js console for document queries, streaming answers, sources, metrics, integrations, and evaluation
- Deterministic evaluation runner for local regression testing

## Setup

From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Expected test result:

```text
40 passed
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

Force the default local orchestrator explicitly:

```powershell
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --orchestrator local
```

Use the optional LangGraph adapter after installing production dependencies and setting configuration:

```powershell
python -m pip install -e ".[production]"
$env:ENABLE_LANGGRAPH = "true"
python -m multi_agent_rag ask "How does RAG reduce hallucination?" --orchestrator langgraph
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
ready: 1/5
local_hybrid_store: ready
qdrant: missing_config
neo4j: missing_config
bge_reranker: missing_config
langgraph: missing_config
```

This is expected without live external services. The default local demo remains fully runnable.

## API Demo

Start the API:

```powershell
python -m uvicorn multi_agent_rag.api.main:app --reload --app-dir src
```

Open:

```text
http://127.0.0.1:8000/docs
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
  "orchestrator": "local"
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
- Switch `Orchestrator` between `Auto` and `Local deterministic` for the local demo.
- Run `Query` mode and inspect the final grounded answer.
- Run `Stream` mode and watch `answer_delta` events render progressively.
- Inspect source snippets and retrieval highlights.
- Check `Integrations` for local and optional production readiness.
- Click `Run Eval` in the `Evaluation` panel and confirm `3/3 passed`.

## Current Limitations

- Local vector retrieval is vector-like lexical scoring, not a learned embedding model.
- Qdrant, Neo4j, reranker, and LangGraph production adapters are readiness boundaries rather than live service implementations.
- PDF ingestion depends on extractable text and does not perform OCR.
- The deterministic judge uses lexical overlap and should be replaced or augmented with an LLM judge for production evaluation.
