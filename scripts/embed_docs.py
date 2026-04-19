from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

from backend.content.catalog import ContentCatalog
from backend.content.manifest import iso_utc_now, refresh_manifest
from backend.core.embeddings import OpenRouterEmbeddings
from backend.core.settings import get_settings
from backend.rag.vectorstore import build_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunk JSONL files into the local Chroma index.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--doc-id", action="append", default=[])
    return parser.parse_args()


def embed_documents(data_root: Path, doc_ids: list[str] | None = None) -> None:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is required to embed documents.")
    catalog = ContentCatalog(data_root)
    embeddings = OpenRouterEmbeddings(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_embedding_model,
        base_url=settings.openrouter_base_url,
        referer=settings.openrouter_referer,
        app_title=settings.openrouter_app_title,
    )
    store = build_vector_store(settings, embeddings)

    target_doc_ids = doc_ids or [doc.doc_id for doc in catalog.list_documents()]
    for doc_id in target_doc_ids:
        records = catalog.load_chunk_records(doc_id)
        if not records:
            continue

        collection = getattr(store, "_collection", None)
        if collection is not None:
            existing = collection.get(where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", []) if existing else []
            if ids:
                store.delete(ids=ids)

        documents = [
            Document(
                page_content=record["text"],
                metadata={
                    "doc_id": record["doc_id"],
                    "chunk_id": record["chunk_id"],
                    "chunk_index": record["chunk_index"],
                    "breadcrumbs": record.get("breadcrumbs", ""),
                    "source_path": record["source_path"],
                    "source_file": record["source_file"],
                    "language": record.get("language", "et"),
                },
            )
            for record in records
        ]
        ids = [record["chunk_id"] for record in records]
        store.add_documents(documents=documents, ids=ids)
        refresh_manifest(
            data_root / doc_id,
            embedding_model=settings.openrouter_embedding_model,
            persist_directory=str(settings.chroma_directory),
            indexed_at=iso_utc_now(),
        )
        print(f"Indexed {doc_id}: {len(records)} chunks")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    data_root = Path(args.data_root) if args.data_root else settings.data_root
    embed_documents(data_root, doc_ids=args.doc_id)


if __name__ == "__main__":
    main()
