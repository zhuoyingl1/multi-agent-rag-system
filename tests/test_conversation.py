from datetime import UTC, datetime

from multi_agent_rag.conversation import contextualize_retrieval_query, recent_history
from multi_agent_rag.persistence import ConversationMessage


def message(index: int, role: str, content: str) -> ConversationMessage:
    return ConversationMessage(f"message-{index}", role, content, datetime.now(UTC))


def test_recent_history_keeps_latest_messages_in_order() -> None:
    messages = [message(index, "user" if index % 2 else "assistant", f"content {index}") for index in range(1, 9)]

    history = recent_history(messages, max_messages=4)

    assert [item["content"] for item in history] == ["content 5", "content 6", "content 7", "content 8"]


def test_vague_follow_up_uses_previous_user_question_for_retrieval() -> None:
    history = [
        {"role": "user", "content": "What are the payment terms?"},
        {"role": "assistant", "content": "The initial payment is due at signing."},
    ]

    query, contextualized = contextualize_retrieval_query("Tell me more about that.", history)

    assert contextualized is True
    assert query == "Previous question: What are the payment terms?\nFollow-up question: Tell me more about that."


def test_explicit_new_question_does_not_use_old_context_for_retrieval() -> None:
    history = [{"role": "user", "content": "What are the payment terms?"}]

    query, contextualized = contextualize_retrieval_query(
        "Explain the confidentiality obligations in Section 8.",
        history,
    )

    assert contextualized is False
    assert query == "Explain the confidentiality obligations in Section 8."


def test_short_but_specific_question_remains_independent() -> None:
    history = [{"role": "user", "content": "What are the confidentiality obligations?"}]

    query, contextualized = contextualize_retrieval_query("Payment terms?", history)

    assert contextualized is False
    assert query == "Payment terms?"
