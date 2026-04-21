from __future__ import annotations

from dataclasses import dataclass
from backend.rag.models import RetrievalExplanation, RetrievalStage, RetrievalStageItem, RetrievedChunk

EXPLANATION_STAGE_LIMIT = 4


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

    def retrieve(self, query: str, *, doc_ids: list[str] | None = None) -> RetrievalResult:
        dense_hits = self.source.retrieve_chunks(
            query,
            doc_ids=doc_ids,
            k=self.settings.retrieval_final_k,
            mode="similarity",
        )
        top_chunks = dense_hits

        return RetrievalResult(
            query=query,
            candidates=dense_hits,
            top_chunks=top_chunks,
            explanation=RetrievalExplanation(
                query_used=query,
                dense_hits=_build_stage(dense_hits),
            ),
            debug={
                "dense_hit_count": len(dense_hits),
                "top_chunk_ids": [chunk.chunk_id for chunk in top_chunks],
            },
        )
