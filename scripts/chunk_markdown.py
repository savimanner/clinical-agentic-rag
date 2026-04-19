from __future__ import annotations

import argparse
from pathlib import Path

from backend.content.chunking import chunk_markdown_document, write_chunk_jsonl
from backend.content.manifest import refresh_manifest
from backend.core.settings import get_settings


def find_markdown_sources(data_root: Path, doc_ids: list[str]) -> list[Path]:
    markdown_paths: list[Path] = []
    guideline_dirs = [data_root / doc_id for doc_id in doc_ids] if doc_ids else sorted(
        path for path in data_root.iterdir() if path.is_dir()
    )
    for guideline_dir in guideline_dirs:
        normalized_dir = guideline_dir / "20_normalized_md"
        canonical_dir = guideline_dir / "10_canonical_md"
        preferred_dir = normalized_dir if any(normalized_dir.glob("*.md")) else canonical_dir
        markdown_paths.extend(sorted(preferred_dir.glob("*.md")))
    return markdown_paths


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Split guideline markdown into chunks and save as JSONL.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--language", default="et")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-chars", type=int, default=settings.chunk_target_chars)
    parser.add_argument("--overlap-chars", type=int, default=settings.chunk_overlap_chars)
    parser.add_argument("--hard-max-chars", type=int, default=settings.chunk_hard_max_chars)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    data_root = Path(args.data_root) if args.data_root else settings.data_root
    markdown_paths = find_markdown_sources(data_root, args.doc_id)
    if not markdown_paths:
        print("No markdown files found to split.")
        return

    for markdown_path in markdown_paths:
        guideline_dir = markdown_path.parents[1]
        output_dir = guideline_dir / "30_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{markdown_path.stem}.jsonl"
        if output_path.exists() and not args.force:
            continue

        text = markdown_path.read_text(encoding="utf-8")
        records = chunk_markdown_document(
            text,
            doc_id=guideline_dir.name,
            source_file=markdown_path.name,
            source_path=str(markdown_path),
            language=args.language,
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
            hard_max_chars=args.hard_max_chars,
        )
        write_chunk_jsonl(records, output_path)
        manifest = refresh_manifest(guideline_dir)
        print(f"Chunked {manifest.doc_id}: {len(records)} chunks -> {output_path}")


if __name__ == "__main__":
    main()
