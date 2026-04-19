from __future__ import annotations

from pydantic import BaseModel, Field

from backend.content.catalog import DocumentSummary
from backend.rag.models import Citation


class ChatRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_doc_ids: list[str] = Field(default_factory=list)
    debug_trace: list[dict] | None = None


class HealthResponse(BaseModel):
    status: str
    openrouter_configured: bool
    index_exists: bool
    documents: int
    indexed_documents: int


LibraryResponse = list[DocumentSummary]
