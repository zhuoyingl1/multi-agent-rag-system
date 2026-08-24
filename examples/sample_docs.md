# Multi-Agent RAG Notes

Retrieval augmented generation reduces hallucination risk by grounding generated answers in retrieved source evidence.

## Hybrid Retrieval

Keyword search is useful for exact terms such as grounding, latency, or source coverage. Vector retrieval helps when a query uses related language instead of exact words. Neo4j can expand entities across related document chunks, while Qdrant can support production vector search.

## Evaluation Metrics

Source coverage measures how many answer claims are supported by retrieved evidence. A grounding judge can flag unsupported claims and provide a score for answer quality.

```python
def grounding_score(supported_claims: int, total_claims: int) -> float:
    return supported_claims / max(total_claims, 1)
```

| Signal | Purpose |
| --- | --- |
| Source coverage | Measures evidence support |
| Latency | Tracks workflow speed |
| Error rate | Tracks failed agents or retrieval failures |
