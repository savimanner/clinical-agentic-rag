from dataclasses import dataclass

from backend.agent.runner import AgentDependencies, AgentRunner
from backend.agent.schemas import AnswerDraft, RewrittenQuery
from backend.content.catalog import DocumentSummary
from backend.core.settings import Settings
from backend.rag.models import RetrievedChunk, RetrievalExplanation, RetrievalStage, RetrievalStageItem
from backend.rag.retrieval import RetrievalResult


class FakeStructuredModel:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, _prompt):
        if self.schema is RewrittenQuery:
            self.parent.rewrite_calls += 1
            self.parent.last_prompt = _prompt
            return self.parent.rewrite_response
        return self.parent.answer_response


class FakeChatModel:
    def __init__(self, rewrite_response, answer_response):
        self.rewrite_response = rewrite_response
        self.answer_response = answer_response
        self.rewrite_calls = 0
        self.last_prompt = None

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
            explanation=make_retrieval_explanation(query, top_chunks),
            debug={
                "dense_hit_count": len(top_chunks),
                "top_chunk_ids": [chunk.chunk_id for chunk in top_chunks],
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


def make_retrieval_explanation(query: str, chunks: list[RetrievedChunk]) -> RetrievalExplanation:
    items = [
        RetrievalStageItem(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            breadcrumbs=chunk.breadcrumbs,
            snippet=chunk.text,
            source_path=chunk.source_path,
            rank=index,
        )
        for index, chunk in enumerate(chunks[:1], start=1)
    ]
    stage = RetrievalStage(total_hits=len(chunks), omitted_hits=max(len(chunks) - len(items), 0), items=items)
    return RetrievalExplanation(
        query_used=query,
        dense_hits=stage,
        final_supporting_chunks=RetrievalStage(),
    )


def test_agent_runner_rewrites_once_and_answers(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="refined query"),
        answer_response=AnswerDraft(
            answer="Bounded answer",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    retrieval_pipeline = FakeRetrievalPipeline(
        [
            [make_chunk("demo-guideline::chunk_0000", "Strong evidence")],
        ]
    )
    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key"),
        catalog=FakeCatalog(),
        retrieval_pipeline=retrieval_pipeline,
        tools=[],
        tool_registry={},
    )
    runner = AgentRunner(deps)

    result = runner.answer_question("Need evidence", debug=True)

    assert result["answer"] == "Bounded answer"
    assert result["used_doc_ids"] == ["demo-guideline"]
    assert fake_model.rewrite_calls == 1
    assert [call["query"] for call in retrieval_pipeline.calls] == ["refined query"]
    assert result["retrieval_explanation"]["query_used"] == "Need evidence"
    assert result["retrieval_explanation"]["refined_question_used"] == "refined query"
    assert result["retrieval_explanation"]["final_supporting_chunks"]["items"][0]["chunk_id"] == (
        "demo-guideline::chunk_0000"
    )
    assert result["retrieval_explanation"]["dense_hits"]["items"][0]["chunk_id"] == "demo-guideline::chunk_0000"
    assert result["retrieval_explanation"]["lexical_hits"]["items"] == []
    assert [entry["step"] for entry in result["debug_trace"]] == [
        "user",
        "rewrite_query",
        "dense_retrieval",
        "generate_answer",
    ]


def test_agent_runner_rewrite_prompt_does_not_force_estonian(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="english retrieval query"),
        answer_response=AnswerDraft(answer="unused"),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    runner = AgentRunner(
        AgentDependencies(
            settings=Settings(openrouter_api_key="test-key"),
            catalog=FakeCatalog(),
            retrieval_pipeline=FakeRetrievalPipeline([[]]),
            tools=[],
            tool_registry={},
        )
    )

    result = runner.answer_question("What is first-line treatment?")

    system_prompt = fake_model.last_prompt[0].content
    assert "Write in Estonian" not in system_prompt
    assert result["answer"] == "I don't know based on the indexed guidelines."


def test_agent_runner_returns_conservative_fallback_without_chunks(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="same question"),
        answer_response=AnswerDraft(answer="unused"),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key"),
        catalog=FakeCatalog(),
        retrieval_pipeline=FakeRetrievalPipeline([[]]),
        tools=[],
        tool_registry={},
    )
    runner = AgentRunner(deps)

    result = runner.answer_question("Need evidence", debug=True)

    assert result["answer"] == "I don't know based on the indexed guidelines."
    assert result["citations"] == []
    assert result["retrieval_explanation"]["final_supporting_chunks"]["items"] == []
    assert [entry["step"] for entry in result["debug_trace"]] == [
        "user",
        "rewrite_query",
        "dense_retrieval",
        "generate_answer",
    ]


def test_agent_runner_uses_bounded_prior_turn_history(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="contraindications refined"),
        answer_response=AnswerDraft(answer="Based on the indexed guidelines, continue the prior plan."),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    retrieval_pipeline = FakeRetrievalPipeline([[make_chunk("demo-guideline::chunk_0000")]])
    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key", agent_history_turn_limit=2),
        catalog=FakeCatalog(),
        retrieval_pipeline=retrieval_pipeline,
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
    assert retrieval_pipeline.calls == [
        {"query": "contraindications refined", "doc_ids": []}
    ]


def test_agent_runner_falls_back_to_top_chunks_when_model_returns_no_citations(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="refined query"),
        answer_response=AnswerDraft(
            answer="Bounded answer",
            cited_chunk_ids=[],
        ),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    retrieval_pipeline = FakeRetrievalPipeline(
        [[make_chunk("demo-guideline::chunk_0000"), make_chunk("demo-guideline::chunk_0001")]]
    )
    runner = AgentRunner(
        AgentDependencies(
            settings=Settings(openrouter_api_key="test-key"),
            catalog=FakeCatalog(),
            retrieval_pipeline=retrieval_pipeline,
            tools=[],
            tool_registry={},
        )
    )

    result = runner.answer_question("Need evidence")

    assert [citation["chunk_id"] for citation in result["citations"]] == [
        "demo-guideline::chunk_0000",
        "demo-guideline::chunk_0001",
    ]
    assert result["retrieval_explanation"]["final_supporting_chunks"]["items"][0]["chunk_id"] == (
        "demo-guideline::chunk_0000"
    )


def test_agent_runner_discards_invalid_citations_and_uses_fallback(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="refined query"),
        answer_response=AnswerDraft(
            answer="Bounded answer",
            cited_chunk_ids=["missing::chunk_0000"],
        ),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    retrieval_pipeline = FakeRetrievalPipeline([[make_chunk("demo-guideline::chunk_0000")]])
    runner = AgentRunner(
        AgentDependencies(
            settings=Settings(openrouter_api_key="test-key"),
            catalog=FakeCatalog(),
            retrieval_pipeline=retrieval_pipeline,
            tools=[],
            tool_registry={},
        )
    )

    result = runner.answer_question("Need evidence")

    assert [citation["chunk_id"] for citation in result["citations"]] == ["demo-guideline::chunk_0000"]
    assert result["used_doc_ids"] == ["demo-guideline"]


def test_agent_runner_returns_conservative_fallback_when_citations_cannot_be_built(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="refined query"),
        answer_response=AnswerDraft(
            answer="This answer should be dropped",
            cited_chunk_ids=["missing::chunk_0000"],
        ),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)
    monkeypatch.setattr("backend.agent.runner.build_citations", lambda *_args, **_kwargs: [])

    retrieval_pipeline = FakeRetrievalPipeline([[make_chunk("demo-guideline::chunk_0000")]])
    runner = AgentRunner(
        AgentDependencies(
            settings=Settings(openrouter_api_key="test-key"),
            catalog=FakeCatalog(),
            retrieval_pipeline=retrieval_pipeline,
            tools=[],
            tool_registry={},
        )
    )

    result = runner.answer_question("Need evidence")

    assert result["answer"] == "I don't know based on the indexed guidelines."
    assert result["citations"] == []
    assert result["used_doc_ids"] == []
    assert result["retrieval_explanation"]["final_supporting_chunks"]["items"] == []


def test_agent_runner_clears_citations_when_model_already_returns_conservative_fallback(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="refined query"),
        answer_response=AnswerDraft(
            answer="I don't know based on the indexed guidelines.",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    retrieval_pipeline = FakeRetrievalPipeline([[make_chunk("demo-guideline::chunk_0000")]])
    runner = AgentRunner(
        AgentDependencies(
            settings=Settings(openrouter_api_key="test-key"),
            catalog=FakeCatalog(),
            retrieval_pipeline=retrieval_pipeline,
            tools=[],
            tool_registry={},
        )
    )

    result = runner.answer_question("Need evidence")

    assert result["answer"] == "I don't know based on the indexed guidelines."
    assert result["citations"] == []
    assert result["used_doc_ids"] == []
    assert result["retrieval_explanation"]["final_supporting_chunks"]["items"] == []


def test_agent_runner_normalizes_non_english_fallback_intent(monkeypatch):
    fake_model = FakeChatModel(
        rewrite_response=RewrittenQuery(query="refined query"),
        answer_response=AnswerDraft(
            answer="Ma ei tea, mida soovitada, sest juhendis ei ole selle kohta infot.",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )
    monkeypatch.setattr("backend.agent.runner.get_chat_model", lambda _settings: fake_model)

    retrieval_pipeline = FakeRetrievalPipeline([[make_chunk("demo-guideline::chunk_0000")]])
    runner = AgentRunner(
        AgentDependencies(
            settings=Settings(openrouter_api_key="test-key"),
            catalog=FakeCatalog(),
            retrieval_pipeline=retrieval_pipeline,
            tools=[],
            tool_registry={},
        )
    )

    result = runner.answer_question("Need evidence")

    assert result["answer"] == "I don't know based on the indexed guidelines."
    assert result["citations"] == []
    assert result["used_doc_ids"] == []
    assert result["retrieval_explanation"]["final_supporting_chunks"]["items"] == []
