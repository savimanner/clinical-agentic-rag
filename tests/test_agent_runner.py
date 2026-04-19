from dataclasses import dataclass

from backend.agent.graph import AgentDependencies
from backend.agent.runner import AgentRunner
from backend.agent.schemas import AnswerDraft, EvidenceGrade
from backend.content.catalog import DocumentSummary
from backend.core.settings import Settings
from backend.rag.models import RetrievedChunk
from backend.rag.retrieval import RetrievalResult


class FakeStructuredModel:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, _prompt):
        if self.schema is EvidenceGrade:
            index = min(self.parent.grade_calls, len(self.parent.grade_responses) - 1)
            self.parent.grade_calls += 1
            return self.parent.grade_responses[index]
        return self.parent.answer_response


class FakeChatModel:
    def __init__(self, grade_responses, answer_response):
        self.grade_responses = grade_responses
        self.answer_response = answer_response
        self.grade_calls = 0

    def with_structured_output(self, schema):
        return FakeStructuredModel(self, schema)


class FakeCatalog:
    def list_documents(self):
        return [
            DocumentSummary(
                doc_id="demo-guideline",
                title="Demo Guideline",
                language="et",
                chunk_count=2,
                indexed=True,
                manifest_path="manifest.json",
            )
        ]


class FakeRetrievalPipeline:
    def __init__(self, top_chunks_per_call):
        self.top_chunks_per_call = top_chunks_per_call
        self.calls = []

    def retrieve(self, query: str, *, doc_ids=None):
        self.calls.append({"query": query, "doc_ids": list(doc_ids or [])})
        index = min(len(self.calls) - 1, len(self.top_chunks_per_call) - 1)
        top_chunks = self.top_chunks_per_call[index]
        return RetrievalResult(
            query=query,
            candidates=top_chunks,
            top_chunks=top_chunks,
            debug={
                "lexical_hit_count": len(top_chunks),
                "dense_hit_count": len(top_chunks),
                "candidate_count": len(top_chunks),
                "top_chunk_ids": [chunk.chunk_id for chunk in top_chunks],
                "rerank_reasoning": "fake",
            },
        )


@dataclass
class FakeDeps:
    settings: Settings
    catalog: FakeCatalog
    retrieval_pipeline: FakeRetrievalPipeline
    tools: list
    tool_registry: dict


def make_chunk(chunk_id: str, text: str = "Evidence") -> RetrievedChunk:
    return RetrievedChunk(
        doc_id="demo-guideline",
        chunk_id=chunk_id,
        chunk_index=int(chunk_id.rsplit("_", 1)[-1]),
        breadcrumbs="Intro",
        text=text,
        source_path="demo.md",
    )


def test_agent_runner_retries_once_then_answers(monkeypatch):
    fake_model = FakeChatModel(
        grade_responses=[
            EvidenceGrade(sufficient=False, reasoning="Need one more retrieval pass", refined_question="refined query"),
            EvidenceGrade(sufficient=True, reasoning="Now enough evidence", cited_chunk_ids=["demo-guideline::chunk_0001"]),
        ],
        answer_response=AnswerDraft(
            answer="Bounded answer",
            cited_chunk_ids=["demo-guideline::chunk_0001"],
        ),
    )
    monkeypatch.setattr("backend.agent.graph.get_chat_model", lambda _settings: fake_model)

    retrieval_pipeline = FakeRetrievalPipeline(
        [
            [make_chunk("demo-guideline::chunk_0000", "Weak evidence")],
            [make_chunk("demo-guideline::chunk_0001", "Strong evidence")],
        ]
    )
    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key", agent_max_iterations=2),
        catalog=FakeCatalog(),
        retrieval_pipeline=retrieval_pipeline,
        tools=[],
        tool_registry={},
    )
    runner = AgentRunner(deps)

    result = runner.answer_question("Need evidence", debug=True)

    assert result["answer"] == "Bounded answer"
    assert result["used_doc_ids"] == ["demo-guideline"]
    assert [call["query"] for call in retrieval_pipeline.calls] == ["Need evidence", "refined query"]
    assert any(entry["step"] == "rewrite_question" for entry in result["debug_trace"])


def test_agent_runner_returns_conservative_fallback_without_chunks(monkeypatch):
    fake_model = FakeChatModel(
        grade_responses=[
            EvidenceGrade(sufficient=False, reasoning="No relevant evidence.", refined_question="same question")
        ],
        answer_response=AnswerDraft(answer="unused"),
    )
    monkeypatch.setattr("backend.agent.graph.get_chat_model", lambda _settings: fake_model)

    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key", agent_max_iterations=1),
        catalog=FakeCatalog(),
        retrieval_pipeline=FakeRetrievalPipeline([[]]),
        tools=[],
        tool_registry={},
    )
    runner = AgentRunner(deps)

    result = runner.answer_question("Need evidence", debug=True)

    assert result["answer"] == "I don't know based on the indexed guidelines."
    assert result["citations"] == []
    assert any(entry["step"] == "grade_evidence" for entry in result["debug_trace"])


def test_agent_runner_uses_bounded_prior_turn_history(monkeypatch):
    fake_model = FakeChatModel(
        grade_responses=[EvidenceGrade(sufficient=True, reasoning="Enough evidence")],
        answer_response=AnswerDraft(answer="Based on the indexed guidelines, continue the prior plan."),
    )
    monkeypatch.setattr("backend.agent.graph.get_chat_model", lambda _settings: fake_model)

    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key", agent_history_turn_limit=2),
        catalog=FakeCatalog(),
        retrieval_pipeline=FakeRetrievalPipeline([[make_chunk("demo-guideline::chunk_0000")]]),
        tools=[],
        tool_registry={},
    )
    runner = AgentRunner(deps)

    result = runner.answer_question(
        "What about contraindications?",
        prior_turns=[
            {"role": "user", "content": "Oldest user turn"},
            {"role": "assistant", "content": "Oldest assistant turn"},
            {"role": "user", "content": "Most recent user turn"},
            {"role": "assistant", "content": "Most recent assistant turn"},
        ],
    )

    assert result["answer"] == "Based on the indexed guidelines, continue the prior plan."
