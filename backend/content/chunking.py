from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field, ValidationError


HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$")
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass
class SplitDoc:
    page_content: str
    metadata: dict[str, str]


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    chunk_index: int
    source_file: str
    source_path: str
    text: str
    search_text: str
    breadcrumbs: str
    language: str = "et"
    version_id: str | None = None
    published_year: int | None = None
    metadata: dict[str, str] | None = None

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        payload = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "language": self.language,
            "breadcrumbs": self.breadcrumbs,
            "text": self.text,
            "search_text": self.search_text,
            "text_hash": self.text_hash,
            "metadata": self.metadata or {},
        }
        if self.version_id:
            payload["version_id"] = self.version_id
        if self.published_year is not None:
            payload["published_year"] = self.published_year
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ChunkArtifactError(ValueError):
    """Raised when a chunk artifact does not match the canonical schema."""


class LexicalPosting(BaseModel):
    chunk_id: str
    tf: int


class LexicalDocument(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    breadcrumbs: str
    source_path: str
    length: int


class LexicalIndexArtifact(BaseModel):
    format: str = "bm25-v1"
    doc_id: str
    chunk_count: int
    avg_document_length: float
    documents: list[LexicalDocument] = Field(default_factory=list)
    postings: dict[str, list[LexicalPosting]] = Field(default_factory=dict)


class ChunkRecordModel(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_index: int
    source_file: str
    source_path: str
    text: str
    search_text: str
    breadcrumbs: str
    language: str = "et"
    version_id: str | None = None
    published_year: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    text_hash: str | None = None


def split_markdown_by_headers(markdown_text: str) -> list[SplitDoc]:
    sections: list[SplitDoc] = []
    headers: dict[str, str | None] = {"Header 1": None, "Header 2": None, "Header 3": None}
    current_lines: list[str] = []

    def flush_section() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            metadata = {key: value for key, value in headers.items() if value}
            sections.append(SplitDoc(page_content=content, metadata=metadata))
        current_lines.clear()

    for line in markdown_text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            flush_section()
            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                headers["Header 1"] = title
                headers["Header 2"] = None
                headers["Header 3"] = None
            elif level == 2:
                headers["Header 2"] = title
                headers["Header 3"] = None
            else:
                headers["Header 3"] = title
            continue
        current_lines.append(line)

    flush_section()
    return sections


def _text_windows(text: str, *, target_chars: int, overlap_chars: int, hard_max_chars: int) -> list[str]:
    clean_text = text.strip()
    if len(clean_text) <= hard_max_chars:
        return [clean_text]

    blocks = [block.strip() for block in re.split(r"\n\s*\n", clean_text) if block.strip()]
    if not blocks:
        blocks = [clean_text]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            flush()
        if len(block) <= hard_max_chars:
            current = block
            continue

        start = 0
        while start < len(block):
            end = min(len(block), start + target_chars)
            window = block[start:end]
            if end < len(block):
                split_at = max(window.rfind("\n"), window.rfind(". "), window.rfind(" "))
                if split_at > target_chars // 2:
                    window = window[:split_at].rstrip()
                    end = start + split_at
            chunks.append(window.strip())
            if end >= len(block):
                break
            start = max(end - overlap_chars, start + 1)

    flush()
    deduped = [chunk for chunk in chunks if chunk]
    return deduped or [clean_text]


def chunk_markdown_document(
    markdown_text: str,
    *,
    doc_id: str,
    source_file: str,
    source_path: str,
    version_id: str | None = None,
    published_year: int | None = None,
    language: str = "et",
    target_chars: int = 1000,
    overlap_chars: int = 120,
    hard_max_chars: int = 1600,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    section_index = 0

    for section in split_markdown_by_headers(markdown_text):
        breadcrumbs = " > ".join(
            value for value in (
                section.metadata.get("Header 1"),
                section.metadata.get("Header 2"),
                section.metadata.get("Header 3"),
            )
            if value
        )
        search_prefix = f"{breadcrumbs}\n" if breadcrumbs else ""
        for piece in _text_windows(
            section.page_content,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            hard_max_chars=hard_max_chars,
        ):
            chunk_id = f"{doc_id}::chunk_{section_index:04d}"
            records.append(
                ChunkRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=section_index,
                    source_file=source_file,
                    source_path=source_path,
                    language=language,
                    version_id=version_id,
                    published_year=published_year,
                    breadcrumbs=breadcrumbs,
                    text=piece,
                    search_text=f"{search_prefix}{piece}" if search_prefix else piece,
                    metadata=dict(section.metadata),
                )
            )
            section_index += 1

    return records


def write_chunk_jsonl(records: list[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(record.to_json() for record in records) + "\n", encoding="utf-8")


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def build_lexical_index(records: list[ChunkRecord]) -> LexicalIndexArtifact:
    if not records:
        raise ValueError("Cannot build a lexical index without chunk records.")

    documents: list[LexicalDocument] = []
    postings: dict[str, list[LexicalPosting]] = {}
    lengths: list[int] = []

    for record in records:
        tokens = _tokens(record.search_text)
        token_count = len(tokens)
        lengths.append(token_count)
        documents.append(
            LexicalDocument(
                chunk_id=record.chunk_id,
                doc_id=record.doc_id,
                chunk_index=record.chunk_index,
                breadcrumbs=record.breadcrumbs,
                source_path=record.source_path,
                length=token_count,
            )
        )
        term_counts: dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
        for term, count in term_counts.items():
            postings.setdefault(term, []).append(LexicalPosting(chunk_id=record.chunk_id, tf=count))

    return LexicalIndexArtifact(
        doc_id=records[0].doc_id,
        chunk_count=len(records),
        avg_document_length=round(mean(lengths) if lengths else 0.0, 3),
        documents=documents,
        postings=postings,
    )


def write_lexical_index(index: LexicalIndexArtifact, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")


def load_lexical_index(index_path: Path) -> LexicalIndexArtifact:
    return LexicalIndexArtifact.model_validate_json(index_path.read_text(encoding="utf-8"))


def _canonical_record_from_payload(payload: dict) -> dict:
    if "properties" in payload or "class" in payload:
        raise ChunkArtifactError(
            "Chunk artifact uses the deprecated schema. Rebuild the corpus with "
            "`./.venv/bin/python scripts/chunk_markdown.py --force` and then "
            "`./.venv/bin/python scripts/embed_docs.py`."
        )
    return payload


def load_chunk_jsonl(chunk_path: Path) -> list[dict]:
    records: list[dict] = []
    for line_number, line in enumerate(chunk_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            canonical = _canonical_record_from_payload(payload)
            record = ChunkRecordModel.model_validate(canonical).model_dump()
        except ChunkArtifactError:
            raise
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ChunkArtifactError(
                f"Invalid chunk artifact at {chunk_path}:{line_number}. "
                "Rebuild the corpus with `./.venv/bin/python scripts/chunk_markdown.py --force`."
            ) from exc
        records.append(record)
    return records
