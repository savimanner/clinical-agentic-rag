from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from backend.content.catalog import ContentCatalog
from backend.core.embeddings import OpenRouterEmbeddings
from backend.core.settings import Settings
from backend.rag.models import LibraryHit, OutlineResponse, RetrievedChunk
from backend.rag.vectorstore import build_chroma_filter, build_vector_store


WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class KnowledgeSource(Protocol):
    def search_library(self, query: str) -> list[LibraryHit]: ...
    def get_document_outline(self, doc_id: str) -> OutlineResponse: ...
    def retrieve_chunks(self, query: str, *, doc_ids: list[str] | None = None, k: int = 5, mode: str = "mmr") -> list[RetrievedChunk]: ...
    def fetch_chunk_neighbors(self, chunk_ids: list[str], *, window: int = 1) -> list[RetrievedChunk]: ...


class LocalCorpusSource:
    def __init__(self, settings: Settings, catalog: ContentCatalog) -> None:
        self.settings = settings
        self.catalog = catalog
        self._vector_store = None

    def _embeddings(self) -> OpenRouterEmbeddings:
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for retrieval and indexing.")
        return OpenRouterEmbeddings(
            api_key=self.settings.openrouter_api_key,
            model=self.settings.openrouter_embedding_model,
            base_url=self.settings.openrouter_base_url,
            referer=self.settings.openrouter_referer,
            app_title=self.settings.openrouter_app_title,
        )

    def _store(self):
        if self._vector_store is None:
            self._vector_store = build_vector_store(self.settings, self._embeddings())
        return self._vector_store

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {match.group(0).lower() for match in WORD_RE.finditer(text)}

    def search_library(self, query: str) -> list[LibraryHit]:
        query_tokens = self._tokens(query)
        hits: list[LibraryHit] = []
        for document in self.catalog.list_documents():
            title_tokens = self._tokens(document.title)
            doc_tokens = self._tokens(document.doc_id)
            overlap = query_tokens & (title_tokens | doc_tokens)
            if not overlap:
                continue
            score = round(len(overlap) / max(len(query_tokens), 1), 3)
            hits.append(
                LibraryHit(
                    doc_id=document.doc_id,
                    title=document.title,
                    score=score,
                    reason=f"Token overlap: {', '.join(sorted(overlap))}",
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:8]

    def get_document_outline(self, doc_id: str) -> OutlineResponse:
        document = self.catalog.get_document(doc_id)
        if not document:
            raise ValueError(f"Unknown document: {doc_id}")
        return OutlineResponse(
            doc_id=document.doc_id,
            title=document.title,
            outline=self.catalog.get_outline(doc_id),
        )

    @staticmethod
    def _doc_to_chunk(document: Document, score: float | None = None) -> RetrievedChunk:
        metadata = document.metadata
        return RetrievedChunk(
            doc_id=str(metadata.get("doc_id", "")),
            chunk_id=str(metadata.get("chunk_id", "")),
            chunk_index=int(metadata.get("chunk_index", 0)),
            breadcrumbs=str(metadata.get("breadcrumbs", "")),
            text=document.page_content,
            source_path=str(metadata.get("source_path", "")),
            score=score,
        )

    def retrieve_chunks(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        k: int = 5,
        mode: str = "mmr",
    ) -> list[RetrievedChunk]:
        store = self._store()
        chroma_filter = build_chroma_filter(doc_ids)
        if mode == "similarity":
            docs_scores = store.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=chroma_filter,
            )
            return [self._doc_to_chunk(doc, score) for doc, score in docs_scores]

        docs = store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=max(self.settings.retrieval_fetch_k, k),
            filter=chroma_filter,
        )
        return [self._doc_to_chunk(doc) for doc in docs]

    @lru_cache(maxsize=128)
    def _chunk_map_for_doc(self, doc_id: str) -> dict[str, dict]:
        records = self.catalog.load_chunk_records(doc_id)
        return {record["chunk_id"]: record for record in records}

    def fetch_chunk_neighbors(self, chunk_ids: list[str], *, window: int = 1) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if "::chunk_" not in chunk_id:
                continue
            doc_id = chunk_id.split("::chunk_", 1)[0]
            records = self.catalog.load_chunk_records(doc_id)
            if not records:
                continue
            by_id = {record["chunk_id"]: idx for idx, record in enumerate(records)}
            center = by_id.get(chunk_id)
            if center is None:
                continue
            for idx in range(max(0, center - window), min(len(records), center + window + 1)):
                record = records[idx]
                if record["chunk_id"] in seen:
                    continue
                seen.add(record["chunk_id"])
                results.append(
                    RetrievedChunk(
                        doc_id=record["doc_id"],
                        chunk_id=record["chunk_id"],
                        chunk_index=int(record["chunk_index"]),
                        breadcrumbs=record.get("breadcrumbs", ""),
                        text=record["text"],
                        source_path=record["source_path"],
                    )
                )
        return results
