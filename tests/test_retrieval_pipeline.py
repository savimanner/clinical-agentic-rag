from __future__ import annotations

from backend.core.settings import Settings
from backend.rag.models import RetrievedChunk
from backend.rag.retrieval import HybridRetrievalPipeline


class FakeSource:
    def retrieve_chunks(self, query: str, *, doc_ids=None, k=5, mode="similarity"):
        assert query == "ibuprofen treatment"
        assert doc_ids == ["demo-guideline"]
        assert k == 2
        assert mode == "similarity"
        return [
            RetrievedChunk(
                doc_id="demo-guideline",
                chunk_id="demo-guideline::chunk_0000",
                chunk_index=0,
                breadcrumbs="Overview",
                text="Pain management overview.",
                source_path="demo.md",
                score=0.8,
            ),
            RetrievedChunk(
                doc_id="demo-guideline",
                chunk_id="demo-guideline::chunk_0002",
                chunk_index=2,
                breadcrumbs="Treatment",
                text="Use ibuprofen for mild pain.",
                source_path="demo.md",
                score=0.7,
            ),
        ]


def test_hybrid_retrieval_pipeline_returns_dense_only_hits():
    pipeline = HybridRetrievalPipeline(
        Settings(openrouter_api_key="test-key", retrieval_final_k=2),
        FakeSource(),
    )

    result = pipeline.retrieve("ibuprofen treatment", doc_ids=["demo-guideline"])

    assert result.debug == {
        "dense_hit_count": 2,
        "top_chunk_ids": ["demo-guideline::chunk_0000", "demo-guideline::chunk_0002"],
    }
    assert result.candidates == result.top_chunks
    assert result.explanation.query_used == "ibuprofen treatment"
    assert result.explanation.lexical_hits.items == []
    assert result.explanation.merged_candidates.items == []
    assert result.explanation.reranked_top_chunks.items == []
    assert [item.chunk_id for item in result.explanation.dense_hits.items] == [
        "demo-guideline::chunk_0000",
        "demo-guideline::chunk_0002",
    ]
