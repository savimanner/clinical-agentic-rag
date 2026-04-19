from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceGrade(BaseModel):
    sufficient: bool
    reasoning: str
    refined_question: str | None = None
    cited_chunk_ids: list[str] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
