from __future__ import annotations

import argparse
from pathlib import Path

from backend.content.chunking import (
    build_lexical_index,
    chunk_markdown_document,
    write_chunk_jsonl,
    write_lexical_index,
)
from backend.content.manifest import refresh_manifest


def resolve_data_root(value: str | None) -> Path:
    return Path(value) if value else Path("data")


def find_markdown_sources(data_root: Path) -> list[tuple[Path, Path]]:
    sources: list[tuple[Path, Path]] = []
    for guideline_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        normalized_dir = guideline_dir / "20_normalized_md"
        canonical_dir = guideline_dir / "10_canonical_md"
        preferred_dir = normalized_dir if any(normalized_dir.glob("*.md")) else canonical_dir
        for markdown_path in sorted(preferred_dir.glob("*.md")):
            sources.append((guideline_dir, markdown_path))
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical chunk and lexical-index artifacts from markdown guidelines."
    )
    parser.add_argument("--data-root", default=None, help="Guideline data root. Defaults to ./data.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing chunk artifacts.")
    parser.add_argument("--target-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--hard-max-chars", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    sources = find_markdown_sources(data_root)
    if not sources:
        print("No markdown files found to chunk.")
        return

    for guideline_dir, markdown_path in sources:
        output_dir = guideline_dir / "30_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = output_dir / f"{markdown_path.stem}.jsonl"
        lexical_path = output_dir / f"{markdown_path.stem}.lexical.json"

        if not args.force and chunk_path.exists() and lexical_path.exists():
            print(f"Skipping {markdown_path.name}: chunk and lexical artifacts already exist.")
            continue

        markdown_text = markdown_path.read_text(encoding="utf-8")
        records = chunk_markdown_document(
            markdown_text,
            doc_id=guideline_dir.name,
            source_file=markdown_path.name,
            source_path=str(markdown_path),
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
            hard_max_chars=args.hard_max_chars,
        )
        lexical_index = build_lexical_index(records)

        write_chunk_jsonl(records, chunk_path)
        write_lexical_index(lexical_index, lexical_path)
        refresh_manifest(guideline_dir)

        print(
            f"Wrote {len(records)} chunks and lexical index for {guideline_dir.name} "
            f"from {markdown_path.name}."
        )


if __name__ == "__main__":
    main()
