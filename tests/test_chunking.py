from backend.content.chunking import chunk_markdown_document


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
