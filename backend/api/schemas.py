from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backend.content.catalog import DocumentSummary
from backend.rag.models import Citation, RetrievalExplanation


class ChatRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_doc_ids: list[str] = Field(default_factory=list)
    retrieval_explanation: RetrievalExplanation | None = None
    debug_trace: list[dict] | None = None


class HealthResponse(BaseModel):
    status: str
    openrouter_configured: bool
    index_exists: bool
    documents: int
    indexed_documents: int


class ThreadScope(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)


class ThreadMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    citations: list[Citation] = Field(default_factory=list)
    used_doc_ids: list[str] = Field(default_factory=list)
    retrieval_explanation: RetrievalExplanation | None = None
    debug_trace: list[dict[str, Any]] | None = None


class ThreadSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    doc_ids: list[str] = Field(default_factory=list)
    message_count: int = 0
    last_message_preview: str | None = None


class ThreadDetail(ThreadSummary):
    messages: list[ThreadMessage] = Field(default_factory=list)


class CreateThreadRequest(BaseModel):
    title: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    scope: ThreadScope | None = None

    def resolved_doc_ids(self) -> list[str]:
        if self.scope is not None:
            return self.scope.doc_ids
        return self.doc_ids


class UpdateThreadRequest(BaseModel):
    title: str | None = None
    doc_ids: list[str] | None = None
    scope: ThreadScope | None = None

    def resolved_doc_ids(self) -> list[str] | None:
        if self.scope is not None:
            return self.scope.doc_ids
        return self.doc_ids


class AppendMessageRequest(BaseModel):
    content: str | None = None
    message: str | None = None
    debug: bool = False

    @model_validator(mode="after")
    def validate_content(self) -> "AppendMessageRequest":
        resolved = (self.content or self.message or "").strip()
        if not resolved:
            raise ValueError("Message content is required.")
        self.content = resolved
        return self


LibraryResponse = list[DocumentSummary]
