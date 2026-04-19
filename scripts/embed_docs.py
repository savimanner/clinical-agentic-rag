from __future__ import annotations

import argparse

import chromadb

from backend.content.catalog import ContentCatalog
from backend.content.manifest import iso_utc_now, refresh_manifest
from backend.core.embeddings import OpenRouterEmbeddings
from backend.core.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed canonical chunk artifacts into the persisted Chroma collection."
    )
    parser.add_argument("--force", action="store_true", help="Replace existing rows for the same chunk ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY in environment or .env file.")

    catalog = ContentCatalog(settings.data_root)
    embeddings = OpenRouterEmbeddings(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_embedding_model,
        base_url=settings.openrouter_base_url,
        referer=settings.openrouter_referer,
        app_title=settings.openrouter_app_title,
    )
    settings.chroma_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_directory))
    collection = client.get_or_create_collection(name=settings.chroma_collection_name)

    indexed_at = iso_utc_now()
    for document in catalog.list_documents():
        records = catalog.load_chunk_records(document.doc_id)
        if not records:
            print(f"Skipping {document.doc_id}: no canonical chunks found.")
            continue

        ids = [record["chunk_id"] for record in records]
        if args.force:
            collection.delete(ids=ids)

        payload_texts = [record["search_text"] for record in records]
        vectors = embeddings.embed_documents(payload_texts)
        metadatas = [
            {
                "doc_id": record["doc_id"],
                "chunk_id": record["chunk_id"],
                "chunk_index": record["chunk_index"],
                "source_file": record["source_file"],
                "source_path": record["source_path"],
                "language": record["language"],
                "breadcrumbs": record["breadcrumbs"],
                "version_id": record.get("version_id") or "",
                "published_year": record.get("published_year") if record.get("published_year") is not None else -1,
            }
            for record in records
        ]
        documents = [record["text"] for record in records]

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
        )
        refresh_manifest(
            settings.data_root / document.doc_id,
            embedding_model=settings.openrouter_embedding_model,
            persist_directory=str(settings.chroma_directory),
            indexed_at=indexed_at,
        )
        print(f"Embedded {len(records)} chunks for {document.doc_id}.")


if __name__ == "__main__":
    main()
