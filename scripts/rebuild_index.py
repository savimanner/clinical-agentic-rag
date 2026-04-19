from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from backend.core.settings import get_settings
from scripts.embed_docs import embed_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear and rebuild the local Chroma index.")
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--skip-clear", action="store_true")
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else settings.data_root
    if not args.skip_clear and settings.chroma_directory.exists():
        shutil.rmtree(settings.chroma_directory)
        print(f"Cleared {settings.chroma_directory}")
    embed_documents(data_root, doc_ids=args.doc_id or None)


if __name__ == "__main__":
    main()
