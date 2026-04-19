from pathlib import Path

from backend.content.catalog import ContentCatalog
from backend.core.settings import Settings
from backend.rag.sources import LocalCorpusSource


class FakeStore:
    def __init__(self):
        self.last_call = None

    def max_marginal_relevance_search(self, query, k, fetch_k, filter=None):
        self.last_call = {"mode": "mmr", "query": query, "k": k, "fetch_k": fetch_k, "filter": filter}
        from langchain_core.documents import Document

        return [
            Document(
                page_content="Chunk text",
                metadata={
                    "doc_id": "demo-guideline",
                    "chunk_id": "demo-guideline::chunk_0000",
                    "chunk_index": 0,
                    "breadcrumbs": "Intro",
                    "source_path": "demo.md",
                },
            )
        ]


def make_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    guideline_dir = data_root / "demo-guideline"
    (guideline_dir / "30_chunks").mkdir(parents=True)
    (guideline_dir / "30_chunks" / "demo.jsonl").write_text(
        "\n".join(
            [
                '{"doc_id":"demo-guideline","chunk_id":"demo-guideline::chunk_0000","chunk_index":0,"source_file":"demo.md","source_path":"demo.md","language":"et","breadcrumbs":"Intro","text":"A","search_text":"Intro\\nA","text_hash":"a","metadata":{}}',
                '{"doc_id":"demo-guideline","chunk_id":"demo-guideline::chunk_0001","chunk_index":1,"source_file":"demo.md","source_path":"demo.md","language":"et","breadcrumbs":"Intro","text":"B","search_text":"Intro\\nB","text_hash":"b","metadata":{}}',
                '{"doc_id":"demo-guideline","chunk_id":"demo-guideline::chunk_0002","chunk_index":2,"source_file":"demo.md","source_path":"demo.md","language":"et","breadcrumbs":"Treatment","text":"C","search_text":"Treatment\\nC","text_hash":"c","metadata":{}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data_root


def test_local_corpus_source_search_neighbors_and_retrieve(tmp_path: Path):
    data_root = make_data_root(tmp_path)
    settings = Settings(data_root=data_root, openrouter_api_key="test-key")
    catalog = ContentCatalog(data_root)
    source = LocalCorpusSource(settings, catalog)
    source._vector_store = FakeStore()

    hits = source.search_library("demo guideline")
    outline = source.get_document_outline("demo-guideline")
    chunks = source.retrieve_chunks("question", doc_ids=["demo-guideline"], k=2)
    neighbors = source.fetch_chunk_neighbors(["demo-guideline::chunk_0001"], window=1)

    assert hits[0].doc_id == "demo-guideline"
    assert "Intro" in outline.outline
    assert chunks[0].chunk_id == "demo-guideline::chunk_0000"
    assert len(neighbors) == 3
    assert source._vector_store.last_call["filter"] == {"doc_id": "demo-guideline"}
