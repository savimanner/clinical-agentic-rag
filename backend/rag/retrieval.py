from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.schemas import RerankSelection
from backend.core.models import get_chat_model
from backend.rag.models import RetrievalExplanation, RetrievalStage, RetrievalStageItem, RetrievedChunk

EXPLANATION_STAGE_LIMIT = 4


def _serialize_candidates(chunks: list[RetrievedChunk], *, limit: int = 32, excerpt_chars: int = 420) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks[:limit], start=1):
        excerpt = chunk.text[:excerpt_chars].strip()
        lines.append(
            f"{index}. [{chunk.chunk_id}] doc={chunk.doc_id} breadcrumbs={chunk.breadcrumbs} "
            f"score={chunk.score if chunk.score is not None else 'n/a'}\n{excerpt}"
        )
    return "\n\n".join(lines)


@dataclass
class RetrievalResult:
    query: str
    candidates: list[RetrievedChunk]
    top_chunks: list[RetrievedChunk]
    explanation: RetrievalExplanation
    debug: dict[str, object]


def _snippet(text: str, *, limit: int = 180) -> str:
    collapsed = " ".join(text.split()).strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3].rstrip()}..."


def _build_stage(
    chunks: list[RetrievedChunk],
    *,
    limit: int = EXPLANATION_STAGE_LIMIT,
    source_modes_by_chunk_id: dict[str, list[str]] | None = None,
    cited_chunk_ids: set[str] | None = None,
) -> RetrievalStage:
    items = [
        RetrievalStageItem(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            breadcrumbs=chunk.breadcrumbs,
            snippet=_snippet(chunk.text),
            source_path=chunk.source_path,
            rank=index,
            score=round(chunk.score, 6) if chunk.score is not None else None,
            source_modes=list(source_modes_by_chunk_id.get(chunk.chunk_id, []))
            if source_modes_by_chunk_id
            else [],
            cited_directly=chunk.chunk_id in cited_chunk_ids if cited_chunk_ids is not None else None,
        )
        for index, chunk in enumerate(chunks[:limit], start=1)
    ]
    return RetrievalStage(
        total_hits=len(chunks),
        omitted_hits=max(len(chunks) - len(items), 0),
        items=items,
    )


class HybridRetrievalPipeline:
    def __init__(self, settings, source) -> None:
        self.settings = settings
        self.source = source

    @staticmethod
    def _rrf_merge(lexical_hits: list[RetrievedChunk], dense_hits: list[RetrievedChunk], *, limit: int) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        rrf_scores: dict[str, float] = {}

        for rank, chunk in enumerate(lexical_hits, start=1):
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (60 + rank))

        for rank, chunk in enumerate(dense_hits, start=1):
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
            elif existing.score is None and chunk.score is not None:
                merged[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (60 + rank))

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        result: list[RetrievedChunk] = []
        for chunk_id in ranked_ids[:limit]:
            chunk = merged[chunk_id].model_copy(deep=True)
            chunk.score = round(rrf_scores[chunk_id], 6)
            result.append(chunk)
        return result

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], str]:
        if len(candidates) <= self.settings.retrieval_final_k:
            return candidates, "Candidate pool is already within the final answer budget."

        model = get_chat_model(self.settings).with_structured_output(RerankSelection)
        prompt = [
            SystemMessage(
                content=(
                    "You are reranking retrieved medical-guideline chunks. "
                    "Pick the chunks that best answer the question while preserving key facts, "
                    "especially exact medical terms, thresholds, and recommendation wording. "
                    "Return chunk ids ordered from most to least useful."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{query}\n\n"
                    f"Candidates:\n{_serialize_candidates(candidates, limit=self.settings.retrieval_candidate_k)}"
                )
            ),
        ]
        selection = model.invoke(prompt)
        selected_ids = selection.ranked_chunk_ids[: self.settings.retrieval_final_k]
        selected = [chunk for chunk in candidates if chunk.chunk_id in selected_ids]
        if not selected:
            selected = candidates[: self.settings.retrieval_final_k]
        else:
            selected.sort(key=lambda chunk: selected_ids.index(chunk.chunk_id))
        return selected, selection.reasoning

    def retrieve(self, query: str, *, doc_ids: list[str] | None = None) -> RetrievalResult:
        lexical_hits = self.source.lexical_search(
            query,
            doc_ids=doc_ids,
            k=self.settings.retrieval_lexical_k,
        )
        dense_hits = self.source.retrieve_chunks(
            query,
            doc_ids=doc_ids,
            k=self.settings.retrieval_candidate_k,
            mode="similarity",
        )
        candidates = self._rrf_merge(
            lexical_hits,
            dense_hits,
            limit=self.settings.retrieval_candidate_k,
        )
        top_chunks, rerank_reasoning = self._rerank(query, candidates)
        source_modes_by_chunk_id: dict[str, list[str]] = {}
        lexical_ids = {chunk.chunk_id for chunk in lexical_hits}
        dense_ids = {chunk.chunk_id for chunk in dense_hits}
        for chunk_id in lexical_ids | dense_ids:
            modes: list[str] = []
            if chunk_id in lexical_ids:
                modes.append("lexical")
            if chunk_id in dense_ids:
                modes.append("dense")
            source_modes_by_chunk_id[chunk_id] = modes

        return RetrievalResult(
            query=query,
            candidates=candidates,
            top_chunks=top_chunks,
            explanation=RetrievalExplanation(
                query_used=query,
                lexical_hits=_build_stage(
                    lexical_hits,
                    source_modes_by_chunk_id={chunk.chunk_id: ["lexical"] for chunk in lexical_hits},
                ),
                dense_hits=_build_stage(
                    dense_hits,
                    source_modes_by_chunk_id={chunk.chunk_id: ["dense"] for chunk in dense_hits},
                ),
                merged_candidates=_build_stage(
                    candidates,
                    source_modes_by_chunk_id=source_modes_by_chunk_id,
                ),
                reranked_top_chunks=_build_stage(
                    top_chunks,
                    source_modes_by_chunk_id=source_modes_by_chunk_id,
                ),
            ),
            debug={
                "lexical_hit_count": len(lexical_hits),
                "dense_hit_count": len(dense_hits),
                "candidate_count": len(candidates),
                "top_chunk_ids": [chunk.chunk_id for chunk in top_chunks],
                "rerank_reasoning": rerank_reasoning,
            },
        )
