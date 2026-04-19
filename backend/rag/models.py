from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    breadcrumbs: str
    snippet: str
    source_path: str


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
