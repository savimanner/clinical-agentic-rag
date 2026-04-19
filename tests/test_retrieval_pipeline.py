from __future__ import annotations

from backend.agent.schemas import RerankSelection
from backend.core.settings import Settings
from backend.rag.models import RetrievedChunk
from backend.rag.retrieval import HybridRetrievalPipeline


class FakeStructuredModel:
    def invoke(self, _prompt):
        return RerankSelection(
            reasoning="The treatment chunk directly answers the question.",
            ranked_chunk_ids=["demo-guideline::chunk_0002", "demo-guideline::chunk_0000"],
        )


class FakeChatModel:
    def with_structured_output(self, _schema):
        return FakeStructuredModel()


class FakeSource:
    def lexical_search(self, query: str, *, doc_ids=None, k=5):
        assert query == "ibuprofen treatment"
        return [
            RetrievedChunk(
                doc_id="demo-guideline",
                chunk_id="demo-guideline::chunk_0002",
                chunk_index=2,
                breadcrumbs="Treatment",
                text="Use ibuprofen for mild pain.",
                source_path="demo.md",
                score=5.0,
            )
        ]

    def retrieve_chunks(self, query: str, *, doc_ids=None, k=5, mode="similarity"):
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


def test_hybrid_retrieval_pipeline_merges_and_reranks(monkeypatch):
    monkeypatch.setattr("backend.rag.retrieval.get_chat_model", lambda _settings: FakeChatModel())
    pipeline = HybridRetrievalPipeline(
        Settings(openrouter_api_key="test-key", retrieval_candidate_k=4, retrieval_final_k=2),
        FakeSource(),
    )

    result = pipeline.retrieve("ibuprofen treatment", doc_ids=["demo-guideline"])

    assert result.debug["lexical_hit_count"] == 1
    assert result.debug["dense_hit_count"] == 2
    assert result.debug["candidate_count"] == 2
    assert [chunk.chunk_id for chunk in result.top_chunks] == [
        "demo-guideline::chunk_0002",
        "demo-guideline::chunk_0000",
    ]
