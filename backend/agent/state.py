from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_trace(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    question: str
    doc_ids: list[str] | None
    working_doc_ids: list[str] | None
    debug: bool
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration_count: int
    tool_call_count: int
    retrieval_attempt_count: int
    retrieval_attempted: bool
    last_tool_names: list[str]
    evidence_ready: bool
    refined_question: str | None
    final_payload: dict[str, Any] | None
    trace: Annotated[list[dict[str, Any]], merge_trace]
