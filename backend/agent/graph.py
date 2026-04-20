from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from backend.agent.schemas import AnswerDraft, EvidenceGrade
from backend.agent.state import AgentState
from backend.core.models import get_chat_model
from backend.rag.citations import build_citations
from backend.rag.models import RetrievalExplanation, RetrievalStage, RetrievalStageItem, RetrievedChunk

FINAL_SUPPORT_LIMIT = 4


@dataclass
class AgentDependencies:
    settings: Any
    catalog: Any
    retrieval_pipeline: Any
    tools: list[Any]
    tool_registry: dict[str, Any]


def _serialize_chunks(chunks: list[RetrievedChunk], limit: int) -> str:
    lines: list[str] = []
    for chunk in chunks[:limit]:
        lines.append(f"[{chunk.chunk_id}] {chunk.doc_id} :: {chunk.breadcrumbs}\n{chunk.text}")
    return "\n\n".join(lines)


def _serialize_conversation_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _state_chunks(state: AgentState) -> list[RetrievedChunk]:
    return [RetrievedChunk.model_validate(payload) for payload in state.get("retrieved_chunks", [])]


def _state_retrieval_explanation(state: AgentState) -> RetrievalExplanation | None:
    payload = state.get("retrieval_explanation")
    if payload is None:
        return None
    return RetrievalExplanation.model_validate(payload)


def _snippet(text: str, *, limit: int = 180) -> str:
    collapsed = " ".join(text.split()).strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3].rstrip()}..."


def _select_chunks_by_ids(chunks: list[RetrievedChunk], chunk_ids: list[str]) -> list[RetrievedChunk]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]


def build_agent_graph(deps: AgentDependencies):
    settings = deps.settings

    def planner_node(state: AgentState) -> dict[str, Any]:
        retrieval_query = state.get("refined_question") or state["question"]
        retrieval = deps.retrieval_pipeline.retrieve(
            retrieval_query,
            doc_ids=state.get("working_doc_ids") or state.get("doc_ids"),
        )
        return {
            "retrieved_chunks": [chunk.model_dump() for chunk in retrieval.top_chunks],
            "retrieval_explanation": retrieval.explanation.model_dump(),
            "retrieval_attempt_count": state.get("retrieval_attempt_count", 0) + 1,
            "trace": [
                {
                    "step": "planner",
                    "query": retrieval_query,
                    "doc_ids": state.get("working_doc_ids") or state.get("doc_ids") or [],
                    **retrieval.debug,
                }
            ],
        }

    def grade_evidence_node(state: AgentState) -> dict[str, Any]:
        chunks = _state_chunks(state)
        conversation_context = _serialize_conversation_history(state.get("conversation_history"))
        conversation_block = f"Prior conversation:\n{conversation_context}\n\n" if conversation_context else ""
        if not chunks:
            next_iteration = state.get("iteration_count", 0) + 1
            return {
                "evidence_ready": False,
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
                    "If evidence is weak or incomplete, provide one refined retrieval question. "
                    "Only cite chunk ids that appear in the provided context."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"{conversation_block}"
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
        return {
            "trace": [{"step": "rewrite_question", "refined_question": refined_question}],
        }

    def generate_answer_node(state: AgentState) -> dict[str, Any]:
        if state.get("final_payload"):
            return {}

        chunks = _state_chunks(state)
        conversation_context = _serialize_conversation_history(state.get("conversation_history"))
        conversation_block = f"Prior conversation:\n{conversation_context}\n\n" if conversation_context else ""
        if not chunks:
            return {
                "final_payload": {
                    "answer": "I don't know based on the indexed guidelines.",
                    "citations": [],
                    "used_doc_ids": [],
                    "retrieval_explanation": _finalize_retrieval_explanation(
                        state,
                        cited_chunk_ids=[],
                        final_chunks=[],
                    ),
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
                    f"{conversation_block}"
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
            "retrieval_explanation": _finalize_retrieval_explanation(
                state,
                cited_chunk_ids=chosen_chunk_ids,
                final_chunks=_select_chunks_by_ids(chunks, chosen_chunk_ids),
            ),
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
        chunks = _state_chunks(state)
        citations = payload.get("citations", [])
        refresh_retrieval_explanation = payload.get("retrieval_explanation") is None
        if not citations and chunks:
            fallback_citations = build_citations(chunks[:3])
            payload["citations"] = [citation.model_dump() for citation in fallback_citations]
            payload["used_doc_ids"] = sorted({citation.doc_id for citation in fallback_citations})
            refresh_retrieval_explanation = True

        if refresh_retrieval_explanation:
            cited_chunk_ids = [citation["chunk_id"] for citation in payload.get("citations", [])]
            payload["retrieval_explanation"] = _finalize_retrieval_explanation(
                state,
                cited_chunk_ids=cited_chunk_ids,
                final_chunks=_select_chunks_by_ids(chunks, cited_chunk_ids),
            )

        if not payload.get("citations") and "guidelines" not in str(payload.get("answer", "")).lower():
            payload["answer"] = "I don't know based on the indexed guidelines."

        if state.get("debug"):
            payload["debug_trace"] = state.get("trace", [])

        return {
            "final_payload": payload,
            "trace": [{"step": "guardrail", "citations": len(payload.get("citations", []))}],
        }

    def after_grading(state: AgentState) -> str:
        if state.get("evidence_ready"):
            return "generate_answer"
        if state.get("iteration_count", 0) >= settings.agent_max_iterations:
            return "generate_answer"
        return "rewrite_question"

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("grade_evidence", grade_evidence_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "grade_evidence")
    graph.add_conditional_edges(
        "grade_evidence",
        after_grading,
        {"rewrite_question": "rewrite_question", "generate_answer": "generate_answer"},
    )
    graph.add_edge("rewrite_question", "planner")
    graph.add_edge("generate_answer", "guardrail")
    graph.add_edge("guardrail", END)
    return graph.compile()


def _finalize_retrieval_explanation(
    state: AgentState,
    *,
    cited_chunk_ids: list[str],
    final_chunks: list[RetrievedChunk],
) -> dict[str, Any] | None:
    explanation = _state_retrieval_explanation(state)
    if explanation is None:
        return None

    if state.get("retrieval_attempt_count", 0) > 1 and explanation.query_used != state["question"]:
        explanation.refined_question_used = explanation.query_used

    cited_chunk_id_set = set(cited_chunk_ids)
    source_modes_by_chunk_id = {
        item.chunk_id: list(item.source_modes) for item in explanation.reranked_top_chunks.items
    }
    explanation.final_supporting_chunks = RetrievalStage(
        total_hits=len(final_chunks),
        omitted_hits=max(len(final_chunks) - min(len(final_chunks), FINAL_SUPPORT_LIMIT), 0),
        items=[
            RetrievalStageItem(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                breadcrumbs=chunk.breadcrumbs,
                snippet=_snippet(chunk.text),
                source_path=chunk.source_path,
                rank=index,
                score=round(chunk.score, 6) if chunk.score is not None else None,
                source_modes=source_modes_by_chunk_id.get(chunk.chunk_id, []),
                cited_directly=chunk.chunk_id in cited_chunk_id_set,
            )
            for index, chunk in enumerate(final_chunks[:FINAL_SUPPORT_LIMIT], start=1)
        ],
    )
    return explanation.model_dump()
