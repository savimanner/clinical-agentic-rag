from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


def pdf_to_markdown(
    file_path_in: str,
    file_path_out: str,
    *,
    model: str = "openai/gpt-4.1-mini",
    api_key_env: str = "OPENROUTER_API_KEY",
    base_url: str = "https://openrouter.ai/api/v1",
    max_pages: Optional[int] = None,
    referer: str = "http://localhost",
    app_title: str = "fastapi-learning-rag",
) -> str:
    """
    Convert a PDF to canonical Markdown using pypdf for text extraction and
    OpenRouter's OpenAI-compatible Responses API for cleanup.
    """
    load_dotenv()

    from os import environ

    token = environ.get(api_key_env)
    if not token:
        raise RuntimeError(f"Missing API key. Set {api_key_env} in environment or .env")

    in_path = Path(file_path_in)
    if not in_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {in_path}")
    if in_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF: {in_path}")

    out_path = Path(file_path_out)
    if out_path.suffix.lower() == ".md":
        final_out_path = out_path
        final_out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        final_out_path = out_path / f"{in_path.stem}.md"

    reader = PdfReader(str(in_path))
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages) if max_pages else total_pages
    page_texts = [(reader.pages[i].extract_text() or "").strip() for i in range(limit)]
    raw_text = "\n\n".join(text for text in page_texts if text)

    if not raw_text.strip():
        raise RuntimeError("No extractable text found in PDF (may be scanned or image-only).")

    instruction = f"""You are producing a canonical Markdown document for downstream
chunking and embedding.

Convert this PDF-extracted text into clean Markdown with headings, lists, and tables.
Preserve the original language. Do not invent content. Keep section hierarchy explicit.

=== PDF TEXT (pypdf) ===
<<<
{raw_text}
>>>
"""

    client = OpenAI(
        api_key=token,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": referer,
            "X-Title": app_title,
        },
    )
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": instruction}],
            }
        ],
    )
    markdown_output = response.output_text.strip()
    final_out_path.write_text(markdown_output + "\n", encoding="utf-8")
    return str(final_out_path)
