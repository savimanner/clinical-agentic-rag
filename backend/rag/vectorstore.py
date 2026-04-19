from __future__ import annotations

from langchain_chroma import Chroma

from backend.core.embeddings import OpenRouterEmbeddings
from backend.core.settings import Settings


def build_vector_store(settings: Settings, embeddings: OpenRouterEmbeddings) -> Chroma:
    settings.chroma_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_directory),
    )


def build_chroma_filter(doc_ids: list[str] | None) -> dict | None:
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}
    return {"doc_id": {"$in": doc_ids}}
