from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    breadcrumbs: str
    snippet: str
    source_path: str


class RetrievalStageItem(BaseModel):
    doc_id: str
    chunk_id: str
    breadcrumbs: str
    snippet: str
    source_path: str
    rank: int | None = None
    score: float | None = None
    source_modes: list[Literal["lexical", "dense"]] = Field(default_factory=list)
    cited_directly: bool | None = None


class RetrievalStage(BaseModel):
    total_hits: int = 0
    omitted_hits: int = 0
    items: list[RetrievalStageItem] = Field(default_factory=list)


class RetrievalExplanation(BaseModel):
    query_used: str
    refined_question_used: str | None = None
    lexical_hits: RetrievalStage = Field(default_factory=RetrievalStage)
    dense_hits: RetrievalStage = Field(default_factory=RetrievalStage)
    merged_candidates: RetrievalStage = Field(default_factory=RetrievalStage)
    reranked_top_chunks: RetrievalStage = Field(default_factory=RetrievalStage)
    final_supporting_chunks: RetrievalStage = Field(default_factory=RetrievalStage)


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_index: int
    breadcrumbs: str
    text: str
    source_path: str
    score: float | None = None


class LibraryHit(BaseModel):
    doc_id: str
    title: str
    score: float
    reason: str


class OutlineResponse(BaseModel):
    doc_id: str
    title: str
    outline: list[str] = Field(default_factory=list)
