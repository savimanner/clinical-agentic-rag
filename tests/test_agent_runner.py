from dataclasses import dataclass

from langchain_core.messages import AIMessage

from backend.agent.graph import AgentDependencies
from backend.agent.runner import AgentRunner
from backend.agent.schemas import AnswerDraft, EvidenceGrade
from backend.content.catalog import DocumentSummary
from backend.core.settings import Settings
from backend.rag.models import LibraryHit, OutlineResponse, RetrievedChunk
from backend.rag.tools import build_rag_tools


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
    def __init__(self, planner_responses, grade_responses, answer_response):
        self.planner_responses = planner_responses
        self.grade_responses = grade_responses
        self.answer_response = answer_response
        self.planner_calls = 0
        self.grade_calls = 0
        self.planner_inputs = []

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema):
        return FakeStructuredModel(self, schema)

    def invoke(self, messages):
        self.planner_inputs.append(messages)
        index = min(self.planner_calls, len(self.planner_responses) - 1)
        self.planner_calls += 1
        return self.planner_responses[index]


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


class FakeSource:
    def search_library(self, query: str):
        return [LibraryHit(doc_id="demo-guideline", title="Demo Guideline", score=1.0, reason=query)]

    def get_document_outline(self, doc_id: str):
        return OutlineResponse(doc_id=doc_id, title="Demo Guideline", outline=["Intro"])

    def retrieve_chunks(self, query: str, *, doc_ids=None, k=5, mode="mmr"):
        return [
            RetrievedChunk(
                doc_id="demo-guideline",
                chunk_id="demo-guideline::chunk_0000",
                chunk_index=0,
                breadcrumbs="Intro",
                text=f"Evidence for {query}",
                source_path="demo.md",
            )
        ]

    def fetch_chunk_neighbors(self, chunk_ids: list[str], *, window: int = 1):
        return self.retrieve_chunks(chunk_ids[0], doc_ids=None, k=window)


@dataclass
class FakeDeps:
    settings: Settings
    catalog: FakeCatalog
    tools: list
    tool_registry: dict


def test_agent_runner_stops_after_bounded_iterations(monkeypatch):
    fake_model = FakeChatModel(
        planner_responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "retrieve_chunks", "args": {"query": "first pass"}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "retrieve_chunks", "args": {"query": "second pass"}, "id": "call-2", "type": "tool_call"}],
            ),
        ],
        grade_responses=[
            EvidenceGrade(sufficient=False, reasoning="Need another pass", refined_question="second pass"),
            EvidenceGrade(sufficient=False, reasoning="Reached the limit", refined_question="final pass"),
        ],
        answer_response=AnswerDraft(
            answer="Bounded answer",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )
    monkeypatch.setattr("backend.agent.graph.get_chat_model", lambda _settings: fake_model)

    source = FakeSource()
    tools, registry = build_rag_tools(source)
    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key", agent_max_iterations=2),
        catalog=FakeCatalog(),
        tools=tools,
        tool_registry=registry,
    )
    runner = AgentRunner(deps)

    result = runner.answer_question("Need evidence", debug=True)

    assert result["answer"] == "Bounded answer"
    assert result["used_doc_ids"] == ["demo-guideline"]
    assert any(entry["step"] == "grade_evidence" for entry in result["debug_trace"])


def test_agent_runner_metadata_then_retrieval_produces_answer(monkeypatch):
    fake_model = FakeChatModel(
        planner_responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_library", "args": {"query": "adult prevention guideline"}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_document_outline", "args": {"doc_id": "demo-guideline"}, "id": "call-2", "type": "tool_call"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "retrieve_chunks", "args": {"query": "prevention steps for 45 year old male", "doc_ids": ["demo-guideline"]}, "id": "call-3", "type": "tool_call"}],
            ),
        ],
        grade_responses=[
            EvidenceGrade(
                sufficient=True,
                reasoning="The retrieved chunk directly answers the question.",
                cited_chunk_ids=["demo-guideline::chunk_0000"],
            )
        ],
        answer_response=AnswerDraft(
            answer="Use the prevention steps from the adult guideline.",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )
    monkeypatch.setattr("backend.agent.graph.get_chat_model", lambda _settings: fake_model)

    source = FakeSource()
    tools, registry = build_rag_tools(source)
    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key", agent_max_iterations=2),
        catalog=FakeCatalog(),
        tools=tools,
        tool_registry=registry,
    )
    runner = AgentRunner(deps)

    result = runner.answer_question("What prevention steps should I do for 45 year old male in my office?", debug=True)

    steps = [entry["step"] for entry in result["debug_trace"]]
    tool_names = [entry["tool_name"] for entry in result["debug_trace"] if entry["step"] == "tool"]

    assert result["answer"] == "Use the prevention steps from the adult guideline."
    assert tool_names == ["search_library", "get_document_outline", "retrieve_chunks"]
    assert "decide_next_step" in steps
    assert any(entry["step"] == "grade_evidence" and entry["sufficient"] for entry in result["debug_trace"])


def test_agent_runner_metadata_does_not_consume_iterations_before_failed_retrieval(monkeypatch):
    fake_model = FakeChatModel(
        planner_responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_library", "args": {"query": "adult prevention guideline"}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "retrieve_chunks", "args": {"query": "no matching evidence"}, "id": "call-2", "type": "tool_call"}],
            ),
            AIMessage(content="I don't know based on the indexed guidelines.", tool_calls=[]),
        ],
        grade_responses=[
            EvidenceGrade(
                sufficient=False,
                reasoning="No relevant chunks were retrieved.",
                refined_question="narrower prevention query",
            )
        ],
        answer_response=AnswerDraft(
            answer="I don't know based on the indexed guidelines.",
            cited_chunk_ids=[],
        ),
    )
    monkeypatch.setattr("backend.agent.graph.get_chat_model", lambda _settings: fake_model)

    class EmptyRetrievalSource(FakeSource):
        def retrieve_chunks(self, query: str, *, doc_ids=None, k=5, mode="mmr"):
            return []

    source = EmptyRetrievalSource()
    tools, registry = build_rag_tools(source)
    deps = AgentDependencies(
        settings=Settings(openrouter_api_key="test-key", agent_max_iterations=1),
        catalog=FakeCatalog(),
        tools=tools,
        tool_registry=registry,
    )
    runner = AgentRunner(deps)

    result = runner.answer_question("Need evidence", debug=True)

    routing_entries = [entry for entry in result["debug_trace"] if entry["step"] == "decide_next_step"]
    assert routing_entries[0]["reason"] == "metadata_only_tools"
    assert routing_entries[1]["reason"] == "retrieval_attempted_without_chunks"
    assert any(entry["step"] == "grade_evidence" for entry in result["debug_trace"])
    assert result["answer"] == "I don't know based on the indexed guidelines."


def test_agent_runner_uses_bounded_prior_turn_history(monkeypatch):
    fake_model = FakeChatModel(
        planner_responses=[AIMessage(content="Based on the indexed guidelines, continue the prior plan.", tool_calls=[])],
        grade_responses=[],
        answer_response=AnswerDraft(answer="unused"),
    )
    monkeypatch.setattr("backend.agent.graph.get_chat_model", lambda _settings: fake_model)

    source = FakeSource()
    tools, registry = build_rag_tools(source)
    deps = AgentDependencies(
        settings=Settings(
            openrouter_api_key="test-key",
            agent_history_turn_limit=2,
        ),
        catalog=FakeCatalog(),
        tools=tools,
        tool_registry=registry,
    )
    runner = AgentRunner(deps)

    result = runner.answer_question(
        "What about contraindications?",
        prior_turns=[
            {"role": "user", "content": "Oldest user turn"},
            {"role": "assistant", "content": "Oldest assistant turn"},
            {"role": "user", "content": "Keep this user turn"},
            {"role": "assistant", "content": "Keep this assistant turn"},
            {"role": "user", "content": "Most recent user turn"},
            {"role": "assistant", "content": "Most recent assistant turn"},
        ],
    )

    planner_messages = fake_model.planner_inputs[0]
    planner_contents = [str(message.content) for message in planner_messages[1:]]

    assert "indexed guidelines" in result["answer"]
    assert planner_contents == [
        "Keep this user turn",
        "Keep this assistant turn",
        "Most recent user turn",
        "Most recent assistant turn",
        "What about contraindications?",
    ]
