from __future__ import annotations

import argparse
from pathlib import Path

from backend.content.manifest import refresh_manifest
from backend.content.normalize import normalize_markdown_text
from backend.core.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize canonical markdown into 20_normalized_md.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    data_root = Path(args.data_root) if args.data_root else settings.data_root

    guideline_dirs = [data_root / doc_id for doc_id in args.doc_id] if args.doc_id else sorted(
        path for path in data_root.iterdir() if path.is_dir()
    )

    for guideline_dir in guideline_dirs:
        canonical_files = sorted((guideline_dir / "10_canonical_md").glob("*.md"))
        if not canonical_files:
            continue
        for canonical_file in canonical_files:
            normalized_dir = guideline_dir / "20_normalized_md"
            normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_file = normalized_dir / canonical_file.name
            if normalized_file.exists() and not args.force:
                continue
            normalized_text = normalize_markdown_text(canonical_file.read_text(encoding="utf-8"))
            normalized_file.write_text(normalized_text, encoding="utf-8")
            refresh_manifest(guideline_dir)
            print(f"Normalized: {canonical_file} -> {normalized_file}")


if __name__ == "__main__":
    main()
