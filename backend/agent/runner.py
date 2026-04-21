from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.schemas import AnswerDraft, RewrittenQuery
from backend.core.models import get_chat_model
from backend.rag.citations import build_citations
from backend.rag.models import RetrievalExplanation, RetrievalStage, RetrievalStageItem, RetrievedChunk

FINAL_SUPPORT_LIMIT = 4
CONSERVATIVE_FALLBACK_ANSWER = "I don't know based on the indexed guidelines."


@dataclass
class AgentDependencies:
    settings: Any
    catalog: Any
    retrieval_pipeline: Any
    tools: list[Any]
    tool_registry: dict[str, Any]


class ConversationTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


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


def _snippet(text: str, *, limit: int = 180) -> str:
    collapsed = " ".join(text.split()).strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3].rstrip()}..."


def _select_chunks_by_ids(chunks: list[RetrievedChunk], chunk_ids: list[str]) -> list[RetrievedChunk]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]


def _is_conservative_fallback_answer(answer_text: str) -> bool:
    normalized = " ".join(answer_text.lower().split())
    if normalized == CONSERVATIVE_FALLBACK_ANSWER.lower():
        return True
    fallback_markers = ("don't know", "do not know", "ei tea")
    evidence_markers = ("guideline", "guidelines", "indexed", "juhend")
    return any(marker in normalized for marker in fallback_markers) and any(
        marker in normalized for marker in evidence_markers
    )


def _finalize_retrieval_explanation(
    explanation: RetrievalExplanation,
    *,
    original_query: str,
    retrieval_query: str,
    cited_chunk_ids: list[str],
    final_chunks: list[RetrievedChunk],
) -> dict[str, Any]:
    result = explanation.model_copy(deep=True)
    result.query_used = original_query
    if retrieval_query != original_query:
        result.refined_question_used = retrieval_query
    else:
        result.refined_question_used = None

    cited_chunk_id_set = set(cited_chunk_ids)
    source_modes_by_chunk_id = {
        item.chunk_id: list(item.source_modes) for item in result.dense_hits.items
    }
    result.final_supporting_chunks = RetrievalStage(
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
    return result.model_dump()


class AgentRunner:
    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    def _rewrite_query(self, question: str, prior_turns: list[ConversationTurn]) -> str:
        conversation_context = _serialize_conversation_history(prior_turns)
        conversation_block = f"Prior conversation:\n{conversation_context}\n\n" if conversation_context else ""
        model = get_chat_model(self.deps.settings).with_structured_output(RewrittenQuery)
        prompt = [
            SystemMessage(
                content=(
                    "Rewrite the user's question into a single retrieval query for searching the local guideline corpus. "
                    "Preserve exact medical meaning, important qualifiers, and drug names. "
                    "Do not answer the question."
                )
            ),
            HumanMessage(
                content=(
                    f"Original question:\n{question}\n\n"
                    f"{conversation_block}"
                    "Return only the rewritten retrieval query."
                )
            ),
        ]
        rewritten = model.invoke(prompt).query.strip()
        return rewritten or question

    @staticmethod
    def _resolve_citations(
        chunks: list[RetrievedChunk],
        cited_chunk_ids: list[str],
    ) -> tuple[list[str], list[Any]]:
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        valid_chunk_ids = [chunk_id for chunk_id in cited_chunk_ids if chunk_id in chunk_ids]
        if valid_chunk_ids:
            citations = build_citations(chunks, chunk_ids=valid_chunk_ids)
            if citations:
                return valid_chunk_ids, citations

        fallback_chunk_ids = [chunk.chunk_id for chunk in chunks[:3]]
        if not fallback_chunk_ids:
            return [], []

        citations = build_citations(chunks, chunk_ids=fallback_chunk_ids)
        if citations:
            return fallback_chunk_ids, citations
        return [], []

    def _generate_answer(
        self,
        question: str,
        prior_turns: list[ConversationTurn],
        chunks: list[RetrievedChunk],
    ) -> AnswerDraft:
        conversation_context = _serialize_conversation_history(prior_turns)
        conversation_block = f"Prior conversation:\n{conversation_context}\n\n" if conversation_context else ""
        model = get_chat_model(self.deps.settings).with_structured_output(AnswerDraft)
        prompt = [
            SystemMessage(
                content=(
                    "Answer the question strictly from the retrieved guideline evidence. "
                    "If the evidence is insufficient, say you don't know based on the indexed guidelines. "
                    "Return chunk ids that directly support the answer."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"{conversation_block}"
                    f"Evidence:\n{_serialize_chunks(chunks, self.deps.settings.debug_context_limit)}"
                )
            ),
        ]
        return model.invoke(prompt)

    def answer_question(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        debug: bool = False,
        prior_turns: list[ConversationTurn] | None = None,
    ) -> dict:
        bounded_turns = list(prior_turns or [])
        max_history_messages = max(self.deps.settings.agent_history_turn_limit * 2, 0)
        if max_history_messages:
            bounded_turns = bounded_turns[-max_history_messages:]
        else:
            bounded_turns = []
        trace: list[dict[str, Any]] = [
            {
                "step": "user",
                "question": question,
                "doc_ids": doc_ids or [],
                "prior_turn_count": len(bounded_turns),
            }
        ]
        retrieval_query = self._rewrite_query(question, bounded_turns)
        trace.append(
            {
                "step": "rewrite_query",
                "query": retrieval_query,
            }
        )

        retrieval = self.deps.retrieval_pipeline.retrieve(
            retrieval_query,
            doc_ids=doc_ids or None,
        )
        trace.append(
            {
                "step": "dense_retrieval",
                "query": retrieval_query,
                "doc_ids": doc_ids or [],
                **retrieval.debug,
            }
        )

        chunks = retrieval.top_chunks
        if not chunks:
            payload = {
                "answer": CONSERVATIVE_FALLBACK_ANSWER,
                "citations": [],
                "used_doc_ids": [],
                "retrieval_explanation": _finalize_retrieval_explanation(
                    retrieval.explanation,
                    original_query=question,
                    retrieval_query=retrieval_query,
                    cited_chunk_ids=[],
                    final_chunks=[],
                ),
            }
            trace.append({"step": "generate_answer", "mode": "fallback-no-context"})
            if debug:
                payload["debug_trace"] = trace
            return payload

        answer = self._generate_answer(question, bounded_turns, chunks)
        chosen_chunk_ids, citations = self._resolve_citations(chunks, answer.cited_chunk_ids)

        used_doc_ids = sorted({citation.doc_id for citation in citations})
        answer_text = answer.answer
        final_chunks = _select_chunks_by_ids(chunks, chosen_chunk_ids)
        if _is_conservative_fallback_answer(answer_text) or not citations:
            answer_text = CONSERVATIVE_FALLBACK_ANSWER
            chosen_chunk_ids = []
            citations = []
            used_doc_ids = []
            final_chunks = []

        payload = {
            "answer": answer_text,
            "citations": [citation.model_dump() for citation in citations],
            "used_doc_ids": used_doc_ids,
            "retrieval_explanation": _finalize_retrieval_explanation(
                retrieval.explanation,
                original_query=question,
                retrieval_query=retrieval_query,
                cited_chunk_ids=chosen_chunk_ids,
                final_chunks=final_chunks,
            ),
        }
        trace.append(
            {
                "step": "generate_answer",
                "cited_chunk_ids": chosen_chunk_ids,
                "used_doc_ids": used_doc_ids,
                "caveats": answer.caveats,
            }
        )
        if debug:
            payload["debug_trace"] = trace
        return payload
