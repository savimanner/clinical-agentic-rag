from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.schemas import RerankSelection
from backend.core.models import get_chat_model
from backend.rag.models import RetrievedChunk


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
    debug: dict[str, object]


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
        return RetrievalResult(
            query=query,
            candidates=candidates,
            top_chunks=top_chunks,
            debug={
                "lexical_hit_count": len(lexical_hits),
                "dense_hit_count": len(dense_hits),
                "candidate_count": len(candidates),
                "top_chunk_ids": [chunk.chunk_id for chunk in top_chunks],
                "rerank_reasoning": rerank_reasoning,
            },
        )
