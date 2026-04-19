from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from backend.content.chunking import load_chunk_jsonl, load_lexical_index
from backend.content.manifest import DocumentManifest, build_manifest, load_manifest


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    language: str
    source_pdf: str | None = None
    primary_markdown: str | None = None
    chunk_file: str | None = None
    lexical_index_file: str | None = None
    chunk_count: int = 0
    indexed: bool = False
    manifest_path: str


class ContentCatalog:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def guideline_dirs(self) -> list[Path]:
        if not self.data_root.exists():
            return []
        return sorted(path for path in self.data_root.iterdir() if path.is_dir())

    def manifest_for_doc(self, doc_id: str) -> DocumentManifest:
        guideline_dir = self.data_root / doc_id
        manifest = load_manifest(guideline_dir)
        return manifest or build_manifest(guideline_dir)

    def list_documents(self) -> list[DocumentSummary]:
        documents: list[DocumentSummary] = []
        for guideline_dir in self.guideline_dirs():
            manifest = load_manifest(guideline_dir) or build_manifest(guideline_dir)
            primary_markdown = (
                manifest.stages["normalized_markdown"].path
                or manifest.stages["canonical_markdown"].path
            )
            chunk_file = manifest.stages["chunk_jsonl"].path
            lexical_index_file = manifest.stages["lexical_index"].path
            documents.append(
                DocumentSummary(
                    doc_id=manifest.doc_id,
                    title=manifest.title,
                    language=manifest.language,
                    source_pdf=manifest.source_pdf,
                    primary_markdown=primary_markdown,
                    chunk_file=chunk_file,
                    lexical_index_file=lexical_index_file,
                    chunk_count=manifest.index.chunk_count,
                    indexed=bool(manifest.index.indexed_at),
                    manifest_path=str(guideline_dir / "manifest.json"),
                )
            )
        return documents

    def get_document(self, doc_id: str) -> DocumentSummary | None:
        for document in self.list_documents():
            if document.doc_id == doc_id:
                return document
        return None

    def load_chunk_records(self, doc_id: str) -> list[dict]:
        summary = self.get_document(doc_id)
        if not summary or not summary.chunk_file:
            return []
        chunk_path = Path(summary.chunk_file)
        if not chunk_path.exists():
            return []
        return load_chunk_jsonl(chunk_path)

    def get_outline(self, doc_id: str) -> list[str]:
        breadcrumbs: list[str] = []
        seen: set[str] = set()
        for record in self.load_chunk_records(doc_id):
            value = record.get("breadcrumbs", "").strip()
            if value and value not in seen:
                breadcrumbs.append(value)
                seen.add(value)
        return breadcrumbs

    def load_lexical_index(self, doc_id: str):
        summary = self.get_document(doc_id)
        if not summary or not summary.lexical_index_file:
            return None
        lexical_path = Path(summary.lexical_index_file)
        if not lexical_path.exists():
            return None
        return load_lexical_index(lexical_path)
