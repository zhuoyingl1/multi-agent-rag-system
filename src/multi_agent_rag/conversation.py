"""Conversation context helpers for multi-turn retrieval and generation."""

from __future__ import annotations

from multi_agent_rag.persistence import ConversationMessage
from multi_agent_rag.retrieval.tokenization import tokenize

FOLLOW_UP_PREFIXES = (
    "and ",
    "can you explain",
    "give me more",
    "how about",
    "tell me more",
    "what about",
)
FOLLOW_UP_TERMS = {"it", "that", "this", "these", "those", "they", "them", "former", "latter"}


def recent_history(
    messages: list[ConversationMessage],
    max_messages: int = 6,
    max_chars: int = 4000,
) -> list[dict[str, str]]:
    """Return a bounded, ordered conversation window for the answer model."""

    selected: list[dict[str, str]] = []
    used_chars = 0
    for message in reversed(messages[-max_messages:]):
        content = message.content.strip()
        if not content:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        bounded = content[-remaining:]
        selected.append({"role": message.role, "content": bounded})
        used_chars += len(bounded)
    return list(reversed(selected))


def contextualize_retrieval_query(query: str, history: list[dict[str, str]]) -> tuple[str, bool]:
    """Add the previous user question when the current query is a vague follow-up."""

    normalized = " ".join(query.split())
    if not history or not _is_follow_up(normalized):
        return normalized, False

    previous_question = next(
        (message["content"] for message in reversed(history) if message.get("role") == "user"),
        "",
    ).strip()
    if not previous_question:
        return normalized, False
    return f"Previous question: {previous_question}\nFollow-up question: {normalized}", True


def _is_follow_up(query: str) -> bool:
    lowered = query.lower()
    tokens = set(tokenize(lowered))
    return lowered.startswith(FOLLOW_UP_PREFIXES) or bool(tokens & FOLLOW_UP_TERMS)
