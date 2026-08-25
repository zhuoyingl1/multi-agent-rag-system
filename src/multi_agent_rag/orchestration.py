"""Workflow selection helpers for local and production orchestration."""

from __future__ import annotations

import importlib.util
import os
from typing import Protocol

from multi_agent_rag.models import WorkflowResult
from multi_agent_rag.retrieval.hybrid import HybridRetriever
from multi_agent_rag.workflow import MultiAgentRAGWorkflow


class WorkflowRunner(Protocol):
    """Shared workflow interface for local and LangGraph orchestrators."""

    def run(self, query: str) -> WorkflowResult:
        """Run a query through the selected workflow."""


def create_workflow(retriever: HybridRetriever, top_k: int = 5, orchestrator: str | None = None) -> WorkflowRunner:
    selected = (orchestrator or os.getenv("RAG_ORCHESTRATOR") or "auto").lower()
    if selected not in {"auto", "local", "langgraph"}:
        raise ValueError("Workflow orchestrator must be one of: auto, local, langgraph")

    should_use_langgraph = selected == "langgraph" or (selected == "auto" and _langgraph_available())
    if should_use_langgraph:
        if _langgraph_available():
            from multi_agent_rag.langgraph_workflow import LangGraphRAGWorkflow

            return LangGraphRAGWorkflow(retriever, top_k=top_k)
        if selected == "langgraph":
            raise RuntimeError("LangGraph orchestrator requested, but the 'langgraph' package is not installed.")

    return MultiAgentRAGWorkflow(retriever, top_k=top_k)


def _langgraph_available() -> bool:
    return importlib.util.find_spec("langgraph") is not None
