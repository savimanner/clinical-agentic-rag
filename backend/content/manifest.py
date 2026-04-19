from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class StageStatus(BaseModel):
    path: str | None = None
    exists: bool = False
    sha256: str | None = None
    updated_at: str | None = None


class IndexStatus(BaseModel):
    vector_store: str = "chroma"
    persist_directory: str | None = None
    embedding_model: str | None = None
    indexed_at: str | None = None
    chunk_count: int = 0


class DocumentManifest(BaseModel):
    doc_id: str
    title: str
    language: str = "et"
    version_id: str | None = None
    published_year: int | None = None
    source_pdf: str | None = None
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    index: IndexStatus = Field(default_factory=IndexStatus)
    updated_at: str | None = None


def manifest_path_for(guideline_dir: Path) -> Path:
    return guideline_dir / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _title_from_doc_id(doc_id: str) -> str:
    return doc_id.replace("-", " ").replace("_", " ").strip().title()


def _stage_from_file(path: Path | None) -> StageStatus:
    if not path or not path.exists():
        return StageStatus(path=str(path) if path else None)
    return StageStatus(
        path=str(path),
        exists=True,
        sha256=sha256_file(path),
        updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(microsecond=0).isoformat(),
    )


def _first_file(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern))
    return files[0] if files else None


def load_manifest(guideline_dir: Path) -> DocumentManifest | None:
    path = manifest_path_for(guideline_dir)
    if not path.exists():
        return None
    return DocumentManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_manifest(guideline_dir: Path) -> DocumentManifest:
    existing = load_manifest(guideline_dir)
    doc_id = guideline_dir.name
    raw_pdf = _first_file(guideline_dir / "00_raw", "*.pdf")
    canonical_md = _first_file(guideline_dir / "10_canonical_md", "*.md")
    normalized_md = _first_file(guideline_dir / "20_normalized_md", "*.md")
    chunk_jsonl = _first_file(guideline_dir / "30_chunks", "*.jsonl")
    lexical_index = _first_file(guideline_dir / "30_chunks", "*.lexical.json")

    manifest = DocumentManifest(
        doc_id=doc_id,
        title=existing.title if existing else _title_from_doc_id(doc_id),
        language=existing.language if existing else "et",
        version_id=existing.version_id if existing else None,
        published_year=existing.published_year if existing else None,
        source_pdf=str(raw_pdf) if raw_pdf else None,
        stages={
            "raw_pdf": _stage_from_file(raw_pdf),
            "canonical_markdown": _stage_from_file(canonical_md),
            "normalized_markdown": _stage_from_file(normalized_md),
            "chunk_jsonl": _stage_from_file(chunk_jsonl),
            "lexical_index": _stage_from_file(lexical_index),
        },
        index=existing.index if existing else IndexStatus(),
        updated_at=iso_utc_now(),
    )

    if chunk_jsonl and chunk_jsonl.exists():
        manifest.index.chunk_count = sum(1 for _ in chunk_jsonl.open(encoding="utf-8"))

    return manifest


def save_manifest(guideline_dir: Path, manifest: DocumentManifest) -> Path:
    path = manifest_path_for(guideline_dir)
    manifest.updated_at = iso_utc_now()
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def refresh_manifest(
    guideline_dir: Path,
    *,
    embedding_model: str | None = None,
    persist_directory: str | None = None,
    indexed_at: str | None = None,
) -> DocumentManifest:
    manifest = build_manifest(guideline_dir)
    if embedding_model is not None:
        manifest.index.embedding_model = embedding_model
    if persist_directory is not None:
        manifest.index.persist_directory = persist_directory
    if indexed_at is not None:
        manifest.index.indexed_at = indexed_at
    save_manifest(guideline_dir, manifest)
    return manifest


def load_manifest_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
