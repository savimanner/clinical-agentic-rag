from __future__ import annotations

from backend.rag.models import Citation, RetrievedChunk


def build_citations(chunks: list[RetrievedChunk], *, chunk_ids: list[str] | None = None) -> list[Citation]:
    allowed = set(chunk_ids or [])
    citations: list[Citation] = []
    for chunk in chunks:
        if allowed and chunk.chunk_id not in allowed:
            continue
        snippet = chunk.text[:240].strip()
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                breadcrumbs=chunk.breadcrumbs,
                snippet=snippet,
                source_path=chunk.source_path,
            )
        )
    return citations
