from __future__ import annotations

import argparse
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from backend.content.manifest import refresh_manifest
from backend.content.pdf_markdown import pdf_to_markdown
from backend.core.settings import get_settings


def slugify(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text or "guideline"


def resolve_data_root(value: str | None) -> Path:
    return Path(value) if value else get_settings().data_root


def collect_tasks_from_structured(data_root: Path, force: bool) -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []
    for pdf_path in sorted(data_root.glob("*/00_raw/*.pdf")):
        guideline_dir = pdf_path.parents[1]
        out_path = guideline_dir / "10_canonical_md" / f"{pdf_path.stem}.md"
        if not force and out_path.exists() and out_path.stat().st_size > 0:
            continue
        tasks.append((pdf_path, out_path))
    return tasks


def collect_tasks_from_flat(input_dir: Path, data_root: Path, force: bool) -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        slug = slugify(pdf_path.stem)
        out_path = data_root / slug / "10_canonical_md" / f"{slug}.md"
        if not force and out_path.exists() and out_path.stat().st_size > 0:
            continue
        tasks.append((pdf_path, out_path))
    return tasks


def convert_one(pdf_path: Path, out_path: Path, *, model: str, max_pages: Optional[int]) -> str:
    settings = get_settings()
    return pdf_to_markdown(
        str(pdf_path),
        str(out_path),
        model=model,
        max_pages=max_pages,
        base_url=settings.openrouter_base_url,
        referer=settings.openrouter_referer,
        app_title=settings.openrouter_app_title,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert guideline PDFs to canonical Markdown.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--model", default=get_settings().openrouter_model)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 4))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    input_dir = Path(args.input_dir) if args.input_dir else None

    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    tasks = (
        collect_tasks_from_flat(input_dir, data_root, args.force)
        if input_dir
        else collect_tasks_from_structured(data_root, args.force)
    )
    if not tasks:
        print("No PDFs to convert.")
        return

    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(convert_one, pdf_path, out_path, model=args.model, max_pages=args.max_pages): (pdf_path, out_path)
            for pdf_path, out_path in tasks
        }
        for future in as_completed(future_map):
            pdf_path, out_path = future_map[future]
            try:
                result = future.result()
                refresh_manifest(out_path.parents[1])
                print(f"OK: {pdf_path.name} -> {result}")
            except Exception as exc:
                failures.append((pdf_path, exc))
                print(f"FAILED: {pdf_path.name} ({exc})")

    if failures:
        raise SystemExit(f"{len(failures)} PDFs failed to convert.")


if __name__ == "__main__":
    main()
