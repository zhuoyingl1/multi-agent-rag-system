import json
from unittest.mock import patch
from urllib.error import URLError

import pytest

from multi_agent_rag.retrieval.embeddings import OllamaEmbeddingService


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_embedding_service_sends_batch_request() -> None:
    service = OllamaEmbeddingService(model_name="nomic-embed-text")
    response = FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    with patch("multi_agent_rag.retrieval.embeddings.urlopen", return_value=response) as request:
        embeddings = service.encode(["first", "second"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    body = json.loads(request.call_args.args[0].data.decode("utf-8"))
    assert body == {"model": "nomic-embed-text", "input": ["first", "second"]}


def test_ollama_embedding_service_reports_unavailable_server() -> None:
    service = OllamaEmbeddingService()

    with patch("multi_agent_rag.retrieval.embeddings.urlopen", side_effect=URLError("offline")):
        with pytest.raises(RuntimeError, match="embedding service is unavailable"):
            service.encode(["text"])
