"""Final answer composer for the local workflow."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from multi_agent_rag.citations import format_evidence_context
from multi_agent_rag.models import AgentResult, JudgeResult, SearchResult


class SummarizerAgent:
    """Compose a readable final answer from specialist findings."""

    def __init__(
        self,
        answer_composer: "AnswerComposer | None" = None,
        require_llm_answer: bool | None = None,
        default_answer_provider: str | None = None,
    ) -> None:
        self.answer_composer = (
            answer_composer if answer_composer is not None else create_answer_composer(default_provider=default_answer_provider)
        )
        self.require_llm_answer = answer_provider_required() if require_llm_answer is None else require_llm_answer
        self.answer_type = "deterministic"
        self.answer_model = "template"
        self.answer_error = ""

    def summarize(
        self,
        query: str,
        agent_results: list[AgentResult],
        judge: JudgeResult,
        sources: list[SearchResult],
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> str:
        successful = [result for result in agent_results if not result.error]
        answer = self._direct_answer(query, successful, sources, on_answer_delta)
        if on_answer_delta is not None and self.answer_type != "llm":
            on_answer_delta(answer)
        return answer

    def _direct_answer(
        self,
        query: str,
        agent_results: list[AgentResult],
        sources: list[SearchResult],
        on_answer_delta: Callable[[str], None] | None,
    ) -> str:
        if not sources and not self.answer_composer:
            if self.require_llm_answer:
                raise RuntimeError(
                    "LLM answer provider unavailable: LLM_ANSWER_PROVIDER=ollama is required for natural-language answers."
                )
            self.answer_type = "deterministic"
            self.answer_model = "template"
            self.answer_error = ""
            return "No sufficiently relevant retrieved evidence was available, so the workflow cannot provide a grounded direct answer."

        if self.answer_composer:
            try:
                if on_answer_delta is None:
                    answer = self.answer_composer.compose(query, agent_results, sources)
                else:
                    answer = self.answer_composer.compose_stream(query, agent_results, sources, on_answer_delta)
                self.answer_type = "llm"
                self.answer_model = self.answer_composer.model
                self.answer_error = ""
                return answer
            except Exception as exc:
                self.answer_type = "deterministic_fallback"
                self.answer_model = self.answer_composer.model
                self.answer_error = _short_error(exc)
                if self.require_llm_answer:
                    raise RuntimeError(f"LLM answer provider unavailable: {self.answer_error}") from exc
        else:
            if self.require_llm_answer:
                self.answer_type = "missing_llm"
                self.answer_model = "none"
                self.answer_error = "LLM_ANSWER_PROVIDER=ollama is required for natural-language answers."
                raise RuntimeError(f"LLM answer provider unavailable: {self.answer_error}")
            self.answer_type = "deterministic"
            self.answer_model = "template"
            self.answer_error = ""

        fallback = self._deterministic_answer(query, agent_results, sources)
        if fallback:
            return fallback

        title = self._best_title(sources)
        matched_terms = self._matched_terms(query, sources)
        if matched_terms:
            term_text = ", ".join(matched_terms[:8])
            return f"The strongest retrieved match is {title}, supported by evidence mentioning {term_text}."
        return f"The strongest retrieved match is {title}, based on the highest-scoring retrieved evidence."

    def _deterministic_answer(self, query: str, agent_results: list[AgentResult], sources: list[SearchResult]) -> str:
        findings = [self._clean_finding(result.content) for result in agent_results if result.content]
        findings = [finding for finding in findings if finding]
        if not findings:
            return ""

        title = self._best_title(sources)
        body = " ".join(findings[:2])
        body = self._bounded_answer(body)
        matched_terms = self._matched_terms(query, sources)
        support = ""
        if matched_terms:
            support = f" This is supported by retrieved evidence mentioning {', '.join(matched_terms[:8])}."
        if self._asks_for_project_match(query):
            if matched_terms:
                return f"The strongest retrieved match is {title}, supported by evidence mentioning {', '.join(matched_terms[:8])}. {body}"
            return f"The strongest retrieved match is {title}. {body}"
        return f"Based on the retrieved evidence, {body}{support}"

    def _clean_finding(self, text: str) -> str:
        compact = " ".join(text.split())
        marker = " agent finding: "
        if marker in compact:
            compact = compact.split(marker, 1)[1].strip()
        return compact

    def _bounded_answer(self, text: str, max_chars: int = 720) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        boundary = compact.rfind(" ", 0, max_chars)
        end = boundary if boundary > max_chars // 2 else max_chars
        return compact[:end].rstrip() + "..."

    def _asks_for_project_match(self, query: str) -> bool:
        lowered = query.lower()
        return "project" in lowered or "resume" in lowered

    def _best_title(self, sources: list[SearchResult]) -> str:
        best = sources[0]
        chunk_text = " ".join(best.chunk.text.split())
        title = best.chunk.metadata.get("title", best.chunk.document_id)
        project_name = self._project_name(chunk_text)
        if project_name:
            return project_name
        return title

    def _project_name(self, text: str) -> str | None:
        marker = "Projects "
        if marker in text:
            after_marker = text.split(marker, 1)[1].strip()
            date_index = self._first_date_index(after_marker)
            if date_index > 0:
                return after_marker[:date_index].strip()

        date_index = self._first_date_index(text)
        if date_index <= 0:
            return None
        before_date = text[:date_index].strip()
        words = before_date.split()
        if not words:
            return None
        return " ".join(words[-6:])

    def _first_date_index(self, text: str) -> int:
        for index in range(max(0, len(text) - 6)):
            window = text[index : index + 7]
            if self._looks_like_month_year_range(window):
                return index
        return -1

    def _looks_like_month_year_range(self, text: str) -> bool:
        if len(text) != 7:
            return False
        return (
            text[0].isdigit()
            and text[1].isdigit()
            and text[2] == "/"
            and text[3].isdigit()
            and text[4].isdigit()
            and text[5].isdigit()
            and text[6].isdigit()
        )

    def _matched_terms(self, query: str, sources: list[SearchResult]) -> list[str]:
        stopwords = {
            "and",
            "or",
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "which",
            "what",
            "how",
            "resume",
            "project",
            "projects",
            "mention",
            "mentions",
        }
        display_names = {
            "rag": "RAG",
            "fastapi": "FastAPI",
            "next.js": "Next.js",
            "next": "Next.js",
            "js": "JavaScript",
            "langgraph": "LangGraph",
            "qdrant": "Qdrant",
            "neo4j": "Neo4j",
        }
        query_terms = [term.strip(" ?.,:;!()[]{}\"'").lower() for term in query.replace("/", " ").split()]
        query_terms = [term for term in query_terms if len(term) > 1 and term not in stopwords]
        source_text = " ".join(source.chunk.text.lower() for source in sources)
        matched = []
        for term in query_terms:
            display = display_names.get(term, term)
            if term in source_text and display not in matched:
                matched.append(display)
        return matched

class AnswerComposer:
    """Protocol-like base class for optional LLM answer composers."""

    model: str

    def compose(self, query: str, agent_results: list[AgentResult], sources: list[SearchResult]) -> str:
        raise NotImplementedError

    def compose_stream(
        self,
        query: str,
        agent_results: list[AgentResult],
        sources: list[SearchResult],
        on_answer_delta: Callable[[str], None],
    ) -> str:
        answer = self.compose(query, agent_results, sources)
        on_answer_delta(answer)
        return answer


class OllamaAnswerComposer(AnswerComposer):
    """Compose answers from retrieved evidence through a local Ollama model."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 60.0,
        max_tokens: int = 220,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def compose(self, query: str, agent_results: list[AgentResult], sources: list[SearchResult]) -> str:
        response = self._post_json("/api/chat", self._payload(query, agent_results, sources, stream=False))
        answer = ((response.get("message") or {}).get("content") or "").strip()
        if not answer:
            raise ValueError("Ollama returned an empty answer.")
        return answer

    def compose_stream(
        self,
        query: str,
        agent_results: list[AgentResult],
        sources: list[SearchResult],
        on_answer_delta: Callable[[str], None],
    ) -> str:
        deltas: list[str] = []
        for response in self._stream_json("/api/chat", self._payload(query, agent_results, sources, stream=True)):
            delta = ((response.get("message") or {}).get("content") or "")
            if delta:
                deltas.append(delta)
                on_answer_delta(delta)
        answer = "".join(deltas).strip()
        if not answer:
            raise ValueError("Ollama returned an empty streamed answer.")
        return answer

    def _payload(
        self,
        query: str,
        agent_results: list[AgentResult],
        sources: list[SearchResult],
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the retrieved evidence. "
                        "Write clear, direct natural language. "
                        "Do not include headings, workflow traces, scores, or source lists. "
                        "Cite every factual claim with one or more provided evidence ids such as [S1]. "
                        "Place citations at the end of the supported sentence and do not discuss citation ids in prose. "
                        "Never invent an evidence id. "
                        "Use one or two short paragraphs. Say when the evidence is insufficient."
                    ),
                },
                {"role": "user", "content": self._prompt(query, agent_results, sources)},
            ],
            "options": {"temperature": 0.2, "num_predict": self.max_tokens},
        }

    def _prompt(self, query: str, agent_results: list[AgentResult], sources: list[SearchResult]) -> str:
        findings = "\n".join(f"- {result.agent_name}: {result.content}" for result in agent_results[:3])
        evidence = format_evidence_context(sources)
        return (
            f"Question:\n{query}\n\n"
            f"Specialist findings:\n{findings or '- No specialist findings.'}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "Write only the final answer and place citations immediately after the claims they support."
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama is not reachable at {self.base_url}. Start Ollama and try again.") from exc

    def _stream_json(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("error"):
                        raise RuntimeError(f"Ollama streaming failed: {item['error']}")
                    yield item
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama is not reachable at {self.base_url}. Start Ollama and try again.") from exc


def create_answer_composer(default_provider: str | None = None) -> AnswerComposer | None:
    """Create an optional LLM answer composer from environment variables."""

    provider = (os.getenv("LLM_ANSWER_PROVIDER") or default_provider or "").lower()
    if provider != "ollama":
        return None
    return OllamaAnswerComposer(
        model=os.getenv("LLM_ANSWER_MODEL", "qwen2.5:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        timeout_seconds=float(os.getenv("LLM_ANSWER_TIMEOUT_SECONDS", "60")),
        max_tokens=int(os.getenv("LLM_ANSWER_MAX_TOKENS", "220")),
    )


def answer_provider_required() -> bool:
    return os.getenv("LLM_ANSWER_REQUIRED", "").lower() in {"1", "true", "yes", "on"}


def _short_error(exc: Exception, max_chars: int = 180) -> str:
    message = str(exc).replace("\n", " ").strip()
    if not message:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message[:max_chars]}"
