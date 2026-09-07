"""Final answer composer for the local workflow."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from multi_agent_rag.models import AgentResult, JudgeResult, SearchResult


class SummarizerAgent:
    """Compose a readable final answer from specialist findings."""

    def __init__(self, answer_composer: "AnswerComposer | None" = None) -> None:
        self.answer_composer = answer_composer if answer_composer is not None else create_answer_composer()
        self.answer_type = "deterministic"
        self.answer_model = "template"
        self.answer_error = ""

    def summarize(self, query: str, agent_results: list[AgentResult], judge: JudgeResult, sources: list[SearchResult]) -> str:
        successful = [result for result in agent_results if not result.error]
        direct_answer = self._direct_answer(query, successful, sources)
        if not successful:
            analysis_lines = ["- The workflow could not produce a grounded finding from retrieved evidence."]
        else:
            analysis_lines = [f"- {result.agent_name}: {result.content}" for result in successful]

        evidence_lines = []
        for source in sources[:4]:
            title = source.chunk.metadata.get("title", source.chunk.document_id)
            snippet = self._focused_snippet(source)
            evidence_lines.append(f"- {title}: {snippet}")
        if not evidence_lines:
            evidence_lines.append("- No retrieved evidence was available.")

        source_titles = self._source_titles(sources)
        source_text = ", ".join(source_titles[:3]) if source_titles else "no retrieved sources"
        unsupported = "; ".join(judge.unsupported_claims) if judge.unsupported_claims else "None"

        return "\n".join(
            [
                f"Question: {query}",
                "",
                "Answer:",
                direct_answer,
                "",
                "Analysis:",
                *analysis_lines,
                "",
                "Evidence:",
                *evidence_lines,
                "",
                f"Grounding score: {judge.score}",
                f"Unsupported claims: {unsupported}",
                f"Sources: {source_text}",
            ]
        )

    def _direct_answer(self, query: str, agent_results: list[AgentResult], sources: list[SearchResult]) -> str:
        if not sources:
            self.answer_type = "deterministic"
            self.answer_model = "template"
            self.answer_error = ""
            return "No sufficiently relevant retrieved evidence was available, so the workflow cannot provide a grounded direct answer."

        if self.answer_composer:
            try:
                answer = self.answer_composer.compose(query, agent_results, sources)
                self.answer_type = "llm"
                self.answer_model = self.answer_composer.model
                self.answer_error = ""
                return answer
            except Exception as exc:
                self.answer_type = "deterministic_fallback"
                self.answer_model = self.answer_composer.model
                self.answer_error = _short_error(exc)
                if answer_provider_required():
                    raise RuntimeError(f"LLM answer provider unavailable: {self.answer_error}") from exc
        else:
            self.answer_type = "deterministic"
            self.answer_model = "template"
            self.answer_error = ""

        title = self._best_title(sources)
        matched_terms = self._matched_terms(query, sources)
        if matched_terms:
            term_text = ", ".join(matched_terms[:8])
            return f"The strongest retrieved match is {title}, supported by evidence mentioning {term_text}."
        return f"The strongest retrieved match is {title}, based on the highest-scoring retrieved evidence."

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

    def _source_titles(self, sources: list[SearchResult]) -> list[str]:
        titles = []
        for source in sources:
            title = source.chunk.metadata.get("title", source.chunk.document_id)
            if title not in titles:
                titles.append(title)
        return titles

    def _focused_snippet(self, source: SearchResult, max_chars: int = 220) -> str:
        compact = " ".join(source.chunk.text.split())
        lowered = compact.lower()
        positions = [lowered.find(term.lower()) for term in source.highlights if term and lowered.find(term.lower()) >= 0]
        if not positions:
            return self._snippet(compact, max_chars=max_chars)
        start = max(0, min(positions) - max_chars // 3)
        end = min(len(compact), start + max_chars)
        snippet = compact[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(compact):
            snippet = snippet.rstrip() + "..."
        return snippet

    def _snippet(self, text: str, max_chars: int = 220) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."


class AnswerComposer:
    """Protocol-like base class for optional LLM answer composers."""

    model: str

    def compose(self, query: str, agent_results: list[AgentResult], sources: list[SearchResult]) -> str:
        raise NotImplementedError


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
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the retrieved evidence. "
                        "Write clear, direct natural language. "
                        "Do not include headings, workflow traces, scores, or source lists. "
                        "Use one or two short paragraphs. Say when the evidence is insufficient."
                    ),
                },
                {"role": "user", "content": self._prompt(query, agent_results, sources)},
            ],
            "options": {"temperature": 0.2, "num_predict": self.max_tokens},
        }
        response = self._post_json("/api/chat", payload)
        answer = ((response.get("message") or {}).get("content") or "").strip()
        if not answer:
            raise ValueError("Ollama returned an empty answer.")
        return answer

    def _prompt(self, query: str, agent_results: list[AgentResult], sources: list[SearchResult]) -> str:
        findings = "\n".join(f"- {result.agent_name}: {result.content}" for result in agent_results[:3])
        evidence = "\n".join(
            f"- {source.chunk.metadata.get('title', source.chunk.document_id)}: {self._bounded_text(source.chunk.text)}"
            for source in sources[:4]
        )
        return (
            f"Question:\n{query}\n\n"
            f"Specialist findings:\n{findings or '- No specialist findings.'}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "Write only the final answer."
        )

    def _bounded_text(self, text: str, max_chars: int = 700) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        boundary = compact.rfind(" ", 0, max_chars)
        return compact[: boundary if boundary > max_chars // 2 else max_chars].rstrip() + "..."

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


def create_answer_composer() -> AnswerComposer | None:
    """Create an optional LLM answer composer from environment variables."""

    provider = os.getenv("LLM_ANSWER_PROVIDER", "").lower()
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
