from __future__ import annotations

from pydantic import BaseModel, Field


class RewrittenQuery(BaseModel):
    query: str


class AnswerDraft(BaseModel):
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class RerankSelection(BaseModel):
    reasoning: str
    ranked_chunk_ids: list[str] = Field(default_factory=list)
