from __future__ import annotations

import json

from langchain_core.tools import tool

from backend.rag.sources import KnowledgeSource


def build_rag_tools(source: KnowledgeSource):
    @tool
    def search_library(query: str) -> str:
        """Find likely local documents before deeper retrieval."""
        hits = source.search_library(query)
        return json.dumps(
            {"tool": "search_library", "results": [hit.model_dump() for hit in hits]},
            ensure_ascii=False,
        )

    @tool
    def get_document_outline(doc_id: str) -> str:
        """Fetch the document outline so retrieval can be scoped to the right section."""
        outline = source.get_document_outline(doc_id)
        return json.dumps(
            {"tool": "get_document_outline", "result": outline.model_dump()},
            ensure_ascii=False,
        )

    @tool
    def retrieve_chunks(
        query: str,
        doc_ids: list[str] | None = None,
        k: int = 5,
        mode: str = "similarity",
    ) -> str:
        """Retrieve semantically relevant chunks from the local corpus."""
        chunks = source.retrieve_chunks(query, doc_ids=doc_ids, k=k, mode=mode)
        return json.dumps(
            {"tool": "retrieve_chunks", "results": [chunk.model_dump() for chunk in chunks]},
            ensure_ascii=False,
        )

    @tool
    def fetch_chunk_neighbors(chunk_ids: list[str], window: int = 1) -> str:
        """Expand context around known chunk ids using adjacent local chunks."""
        chunks = source.fetch_chunk_neighbors(chunk_ids, window=window)
        return json.dumps(
            {"tool": "fetch_chunk_neighbors", "results": [chunk.model_dump() for chunk in chunks]},
            ensure_ascii=False,
        )

    tools = [search_library, get_document_outline, retrieve_chunks, fetch_chunk_neighbors]
    return tools, {tool_.name: tool_ for tool_ in tools}
