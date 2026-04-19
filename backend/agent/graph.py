from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from backend.agent.schemas import AnswerDraft, EvidenceGrade
from backend.agent.state import AgentState
from backend.core.models import get_chat_model
from backend.rag.citations import build_citations
from backend.rag.models import RetrievedChunk

METADATA_TOOLS = {"search_library", "get_document_outline"}
EVIDENCE_TOOLS = {"retrieve_chunks", "fetch_chunk_neighbors"}


@dataclass
class AgentDependencies:
    settings: Any
    catalog: Any
    tools: list[Any]
    tool_registry: dict[str, Any]


def _safe_json_loads(value: str) -> dict[str, Any] | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _tool_payloads(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            payload = _safe_json_loads(str(message.content))
            if payload:
                payloads.append(payload)
    return payloads


def _retrieved_chunks(messages: list[BaseMessage]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    seen: set[str] = set()
    for payload in _tool_payloads(messages):
        if payload.get("tool") not in {"retrieve_chunks", "fetch_chunk_neighbors"}:
            continue
        for item in payload.get("results", []):
            chunk = RetrievedChunk.model_validate(item)
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


def _serialize_chunks(chunks: list[RetrievedChunk], limit: int) -> str:
    lines: list[str] = []
    for chunk in chunks[:limit]:
        lines.append(
            f"[{chunk.chunk_id}] {chunk.doc_id} :: {chunk.breadcrumbs}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def _latest_tool_payload(messages: list[BaseMessage], tool_name: str) -> dict[str, Any] | None:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        payload = _safe_json_loads(str(message.content))
        if payload and payload.get("tool") == tool_name:
            return payload
    return None


def _derive_working_doc_ids(messages: list[BaseMessage], current_doc_ids: list[str] | None) -> list[str] | None:
    if current_doc_ids:
        return current_doc_ids
    payload = _latest_tool_payload(messages, "search_library")
    if not payload:
        return None
    results = payload.get("results", [])
    if not results:
        return None
    top = results[0]
    score = float(top.get("score", 0))
    if score < 0.6:
        return None
    doc_id = top.get("doc_id")
    return [doc_id] if doc_id else None


def build_agent_graph(deps: AgentDependencies):
    settings = deps.settings

    def planner_node(state: AgentState) -> dict[str, Any]:
        model = get_chat_model(settings).bind_tools(deps.tools)
        documents = deps.catalog.list_documents()
        catalog_summary = "\n".join(
            f"- {document.doc_id}: {document.title} (indexed={document.indexed}, chunks={document.chunk_count})"
            for document in documents[:50]
        )
        working_doc_ids = state.get("working_doc_ids") or state.get("doc_ids")
        scope = (
            f"Current working doc scope: {', '.join(working_doc_ids or [])}"
            if working_doc_ids
            else "No doc filter was provided."
        )
        system_prompt = SystemMessage(
            content=(
                "You are a local medical-guideline research agent. "
                "You only have access to the curated local corpus through tools. "
                "For corpus questions, break the task into up to 3 subquestions internally and use tools iteratively. "
                "Use `search_library` when document choice is unclear, `get_document_outline` to inspect structure, "
                "`retrieve_chunks` for semantic retrieval, and `fetch_chunk_neighbors` to expand local context. "
                "After `search_library`, the next step should usually be `get_document_outline` or `retrieve_chunks`. "
                "After `get_document_outline`, the next step should usually be `retrieve_chunks`. "
                "Avoid stopping before at least one retrieval attempt for corpus questions. "
                "If the user is only greeting or asking meta questions, answer directly without tools. "
                "Never claim facts that are not grounded in tool results.\n\n"
                f"{scope}\nAvailable local documents:\n{catalog_summary}"
            )
        )
        response = model.invoke([system_prompt, *list(state["messages"])])
        updates: dict[str, Any] = {
            "messages": [response],
            "trace": [
                {
                    "step": "planner",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
            ],
        }
        if not response.tool_calls:
            updates["final_payload"] = {
                "answer": response.content or "I don't know based on the indexed guidelines.",
                "citations": [],
                "used_doc_ids": state.get("doc_ids") or [],
            }
        return updates

    def execute_tools_node(state: AgentState) -> dict[str, Any]:
        ai_message = _last_ai_message(list(state["messages"]))
        if not ai_message or not ai_message.tool_calls:
            return {}

        tool_messages: list[ToolMessage] = []
        trace: list[dict[str, Any]] = []
        remaining = max(settings.agent_max_tool_calls - state.get("tool_call_count", 0), 0)
        allowed_calls = ai_message.tool_calls[:remaining]
        last_tool_names = [tool_call["name"] for tool_call in allowed_calls]
        retrieval_attempted = any(tool_name in EVIDENCE_TOOLS for tool_name in last_tool_names)

        for tool_call in allowed_calls:
            tool = deps.tool_registry[tool_call["name"]]
            try:
                result = tool.invoke(tool_call["args"])
            except Exception as exc:  # pragma: no cover - defensive runtime path
                result = json.dumps(
                    {"tool": tool_call["name"], "error": str(exc), "results": []},
                    ensure_ascii=False,
                )
            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
            trace.append(
                {
                    "step": "tool",
                    "tool_name": tool_call["name"],
                    "args": tool_call["args"],
                    "result_preview": str(result)[:500],
                }
            )

        working_doc_ids = _derive_working_doc_ids(
            list(state["messages"]) + tool_messages,
            state.get("doc_ids"),
        )
        return {
            "messages": tool_messages,
            "tool_call_count": state.get("tool_call_count", 0) + len(allowed_calls),
            "retrieval_attempted": retrieval_attempted,
            "retrieval_attempt_count": state.get("retrieval_attempt_count", 0) + (1 if retrieval_attempted else 0),
            "last_tool_names": last_tool_names,
            "working_doc_ids": working_doc_ids,
            "trace": trace,
        }

    def route_after_tools_node(state: AgentState) -> dict[str, Any]:
        chunks = _retrieved_chunks(list(state["messages"]))
        last_tool_names = state.get("last_tool_names", [])
        if chunks:
            route = "grade_evidence"
            reason = "retrieved_chunks_available"
        elif state.get("retrieval_attempted"):
            route = "grade_evidence"
            reason = "retrieval_attempted_without_chunks"
        elif last_tool_names and all(tool_name in METADATA_TOOLS for tool_name in last_tool_names):
            route = "planner"
            reason = "metadata_only_tools"
        elif state.get("tool_call_count", 0) >= settings.agent_max_tool_calls:
            route = "generate_answer"
            reason = "tool_budget_exhausted"
        else:
            route = "planner"
            reason = "continue_planning"
        return {
            "trace": [
                {
                    "step": "decide_next_step",
                    "route": route,
                    "reason": reason,
                    "last_tool_names": last_tool_names,
                    "retrieved_chunk_count": len(chunks),
                    "working_doc_ids": state.get("working_doc_ids") or [],
                }
            ]
        }

    def grade_evidence_node(state: AgentState) -> dict[str, Any]:
        chunks = _retrieved_chunks(list(state["messages"]))
        if not chunks:
            next_iteration = state.get("iteration_count", 0)
            if state.get("retrieval_attempted"):
                next_iteration += 1
            return {
                "evidence_ready": next_iteration >= settings.agent_max_iterations,
                "iteration_count": next_iteration,
                "refined_question": state["question"],
                "trace": [
                    {
                        "step": "grade_evidence",
                        "reasoning": "No retrieved chunks were available to grade.",
                        "sufficient": False,
                    }
                ],
            }

        model = get_chat_model(settings).with_structured_output(EvidenceGrade)
        prompt = [
            SystemMessage(
                content=(
                    "You are grading whether the retrieved local-corpus evidence is sufficient to answer the question. "
                    "Return `sufficient=true` only if the current evidence is enough for a grounded answer. "
                    "If evidence is weak or incomplete, provide a refined retrieval question. "
                    "Only cite chunk ids that appear in the provided context."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Retrieved evidence:\n{_serialize_chunks(chunks, settings.debug_context_limit)}"
                )
            ),
        ]
        grade = model.invoke(prompt)
        next_iteration = state.get("iteration_count", 0)
        if not grade.sufficient:
            next_iteration += 1
        return {
            "evidence_ready": grade.sufficient,
            "iteration_count": next_iteration,
            "refined_question": grade.refined_question,
            "trace": [
                {
                    "step": "grade_evidence",
                    "reasoning": grade.reasoning,
                    "sufficient": grade.sufficient,
                    "refined_question": grade.refined_question,
                    "cited_chunk_ids": grade.cited_chunk_ids,
                }
            ],
        }

    def rewrite_question_node(state: AgentState) -> dict[str, Any]:
        refined_question = state.get("refined_question") or state["question"]
        message = HumanMessage(
            content=(
                "Continue the research loop. Focus retrieval on this refined question:\n"
                f"{refined_question}\n"
                "Use tools again if you still need evidence."
            )
        )
        return {
            "messages": [message],
            "retrieval_attempted": False,
            "last_tool_names": [],
            "trace": [{"step": "rewrite_question", "refined_question": refined_question}],
        }

    def generate_answer_node(state: AgentState) -> dict[str, Any]:
        if state.get("final_payload"):
            return {}

        chunks = _retrieved_chunks(list(state["messages"]))
        if not chunks:
            return {
                "final_payload": {
                    "answer": "I don't know based on the indexed guidelines.",
                    "citations": [],
                    "used_doc_ids": [],
                },
                "trace": [{"step": "generate_answer", "mode": "fallback-no-context"}],
            }

        model = get_chat_model(settings).with_structured_output(AnswerDraft)
        prompt = [
            SystemMessage(
                content=(
                    "Answer the question strictly from the retrieved guideline evidence. "
                    "If the evidence is still insufficient, say you don't know based on the indexed guidelines. "
                    "Return chunk ids that directly support the answer."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Evidence:\n{_serialize_chunks(chunks, settings.debug_context_limit)}"
                )
            ),
        ]
        answer = model.invoke(prompt)
        chosen_chunk_ids = answer.cited_chunk_ids or [chunk.chunk_id for chunk in chunks[:3]]
        citations = build_citations(chunks, chunk_ids=chosen_chunk_ids)
        used_doc_ids = sorted({citation.doc_id for citation in citations})
        payload = {
            "answer": answer.answer,
            "citations": [citation.model_dump() for citation in citations],
            "used_doc_ids": used_doc_ids,
        }
        return {
            "final_payload": payload,
            "trace": [
                {
                    "step": "generate_answer",
                    "cited_chunk_ids": chosen_chunk_ids,
                    "used_doc_ids": used_doc_ids,
                    "caveats": answer.caveats,
                }
            ],
        }

    def guardrail_node(state: AgentState) -> dict[str, Any]:
        payload = dict(state.get("final_payload") or {})
        citations = payload.get("citations", [])
        if not citations and _retrieved_chunks(list(state["messages"])):
            fallback_citations = build_citations(_retrieved_chunks(list(state["messages"]))[:3])
            payload["citations"] = [citation.model_dump() for citation in fallback_citations]
            payload["used_doc_ids"] = sorted({citation.doc_id for citation in fallback_citations})

        if not payload.get("citations") and "guidelines" not in str(payload.get("answer", "")).lower():
            payload["answer"] = "I don't know based on the indexed guidelines."

        if state.get("debug"):
            payload["debug_trace"] = state.get("trace", [])

        return {
            "final_payload": payload,
            "trace": [{"step": "guardrail", "citations": len(payload.get("citations", []))}],
        }

    def after_planner(state: AgentState) -> str:
        if state.get("final_payload"):
            return "guardrail"
        ai_message = _last_ai_message(list(state["messages"]))
        if ai_message and ai_message.tool_calls:
            return "execute_tools"
        return "generate_answer"

    def after_grading(state: AgentState) -> str:
        if state.get("evidence_ready"):
            return "generate_answer"
        if state.get("iteration_count", 0) >= settings.agent_max_iterations:
            return "generate_answer"
        if state.get("tool_call_count", 0) >= settings.agent_max_tool_calls:
            return "generate_answer"
        return "rewrite_question"

    def after_tool_routing(state: AgentState) -> str:
        trace = state.get("trace", [])
        for entry in reversed(trace):
            if entry.get("step") == "decide_next_step":
                return str(entry["route"])
        return "planner"

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("decide_next_step", route_after_tools_node)
    graph.add_node("grade_evidence", grade_evidence_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_planner, {"execute_tools": "execute_tools", "generate_answer": "generate_answer", "guardrail": "guardrail"})
    graph.add_edge("execute_tools", "decide_next_step")
    graph.add_conditional_edges(
        "decide_next_step",
        after_tool_routing,
        {
            "planner": "planner",
            "grade_evidence": "grade_evidence",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges("grade_evidence", after_grading, {"rewrite_question": "rewrite_question", "generate_answer": "generate_answer"})
    graph.add_edge("rewrite_question", "planner")
    graph.add_edge("generate_answer", "guardrail")
    graph.add_edge("guardrail", END)
    return graph.compile()
