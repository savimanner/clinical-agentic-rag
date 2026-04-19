from pathlib import Path

from backend.content.catalog import ContentCatalog
from backend.content.manifest import build_manifest, save_manifest


def test_manifest_and_catalog_reflect_local_guideline_layout(tmp_path: Path):
    data_root = tmp_path / "data"
    guideline_dir = data_root / "demo-guideline"
    (guideline_dir / "00_raw").mkdir(parents=True)
    (guideline_dir / "10_canonical_md").mkdir()
    (guideline_dir / "30_chunks").mkdir()

    (guideline_dir / "00_raw" / "demo.pdf").write_bytes(b"%PDF-1.7 demo")
    (guideline_dir / "10_canonical_md" / "demo.md").write_text("# Demo\n", encoding="utf-8")
    (guideline_dir / "30_chunks" / "demo.jsonl").write_text(
        '{"doc_id":"demo-guideline","chunk_id":"demo-guideline::chunk_0000","chunk_index":0,'
        '"source_file":"demo.md","source_path":"demo.md","language":"et","breadcrumbs":"Demo",'
        '"text":"Hello","search_text":"Demo\\nHello","text_hash":"abc","metadata":{}}\n',
        encoding="utf-8",
    )

    manifest = build_manifest(guideline_dir)
    save_manifest(guideline_dir, manifest)

    catalog = ContentCatalog(data_root)
    documents = catalog.list_documents()

    assert manifest.doc_id == "demo-guideline"
    assert manifest.index.chunk_count == 1
    assert len(documents) == 1
    assert documents[0].doc_id == "demo-guideline"
    assert documents[0].chunk_count == 1
