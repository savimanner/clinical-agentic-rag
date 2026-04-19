from __future__ import annotations

import re


HEADING_RE = re.compile(r"^(#{1,6})([^\s#])")


def normalize_markdown_text(markdown_text: str) -> str:
    """Apply light normalization without changing document meaning."""
    normalized_lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        line = line.replace("\ufeff", "").replace("\x0c", "")
        line = HEADING_RE.sub(r"\1 \2", line)
        normalized_lines.append(line)

    normalized = "\n".join(normalized_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip() + "\n"
