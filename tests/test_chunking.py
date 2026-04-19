from pathlib import Path

import pytest

from backend.content.chunking import (
    ChunkArtifactError,
    build_lexical_index,
    chunk_markdown_document,
    load_chunk_jsonl,
)


def test_chunk_markdown_document_splits_large_sections_and_preserves_breadcrumbs():
    large_paragraph = "Sentence. " * 500
    markdown = f"""# Guideline
## Diagnosis
{large_paragraph}
"""

    chunks = chunk_markdown_document(
        markdown,
        doc_id="demo-guideline",
        source_file="demo.md",
        source_path="data/demo-guideline/20_normalized_md/demo.md",
        target_chars=500,
        overlap_chars=50,
        hard_max_chars=700,
    )

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert all(chunk.breadcrumbs == "Guideline > Diagnosis" for chunk in chunks)
    assert chunks[0].chunk_id == "demo-guideline::chunk_0000"


def test_chunk_markdown_document_builds_lexical_index():
    markdown = """# Guideline
## Treatment
Use ibuprofen for mild pain.
"""

    chunks = chunk_markdown_document(
        markdown,
        doc_id="demo-guideline",
        source_file="demo.md",
        source_path="data/demo-guideline/20_normalized_md/demo.md",
    )
    lexical_index = build_lexical_index(chunks)

    assert lexical_index.doc_id == "demo-guideline"
    assert lexical_index.chunk_count == 1
    assert lexical_index.documents[0].chunk_id == "demo-guideline::chunk_0000"
    assert "ibuprofen" in lexical_index.postings


def test_load_chunk_jsonl_fails_fast_for_deprecated_schema(tmp_path: Path):
    chunk_path = tmp_path / "deprecated.jsonl"
    chunk_path.write_text(
        '{"class":"demo-guideline","properties":{"chunk_id":0,"text":"Deprecated","search_text":"Deprecated"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ChunkArtifactError) as exc:
        load_chunk_jsonl(chunk_path)

    assert "deprecated schema" in str(exc.value)
    assert "scripts/chunk_markdown.py --force" in str(exc.value)
