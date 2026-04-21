from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from backend.agent.runner import AgentDependencies, AgentRunner
from backend.agent.schemas import AnswerDraft, RewrittenQuery
from backend.api.app import create_app
from backend.content.catalog import DocumentSummary
from backend.core.settings import Settings
from backend.rag.models import RetrievedChunk, RetrievalExplanation, RetrievalStage, RetrievalStageItem
from backend.rag.retrieval import RetrievalResult
from backend.threads import LocalThreadStore, ThreadService


class FakeCatalog:
    def list_documents(self):
        return [
            DocumentSummary(
                doc_id="demo-guideline",
                title="Demo Guideline",
                language="et",
                chunk_count=3,
                indexed=True,
                manifest_path="data/demo-guideline/manifest.json",
            )
        ]

    def get_document(self, doc_id: str):
        for document in self.list_documents():
            if document.doc_id == doc_id:
                return document
        return None


def make_retrieval_explanation(query_used: str = "What is this?") -> RetrievalExplanation:
    item = RetrievalStageItem(
        doc_id="demo-guideline",
        chunk_id="demo-guideline::chunk_0000",
        breadcrumbs="Intro",
        snippet="Demo snippet",
        source_path="demo.md",
        rank=1,
        score=1.0,
    )
    stage = RetrievalStage(total_hits=1, omitted_hits=0, items=[item])
    return RetrievalExplanation(
        query_used=query_used,
        dense_hits=stage,
        final_supporting_chunks=stage,
    )


@dataclass
class FakeAgent:
    calls: list[dict] = field(default_factory=list)
    error: Exception | None = None

    def answer_question(self, question: str, *, doc_ids=None, debug=False, prior_turns=None):
        if self.error is not None:
            raise self.error
        self.calls.append(
            {
                "question": question,
                "doc_ids": doc_ids,
                "debug": debug,
                "prior_turns": list(prior_turns or []),
            }
        )
        return {
            "answer": f"Answered: {question}",
            "citations": [
                {
                    "doc_id": "demo-guideline",
                    "chunk_id": "demo-guideline::chunk_0000",
                    "breadcrumbs": "Intro",
                    "snippet": "Demo snippet",
                    "source_path": "demo.md",
                }
            ],
            "used_doc_ids": ["demo-guideline"],
            "retrieval_explanation": make_retrieval_explanation(question).model_dump(),
            "debug_trace": [{"step": "dense_retrieval"}] if debug else None,
        }


@dataclass
class FakeRuntime:
    settings: Settings
    catalog: FakeCatalog
    agent: FakeAgent
    thread_store: LocalThreadStore
    thread_service: ThreadService


class StructuredResponder:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, _prompt):
        if self.schema is RewrittenQuery:
            return self.parent.rewrite_response
        return self.parent.answer_response


class FakeChatModel:
    def __init__(self, rewrite_response, answer_response):
        self.rewrite_response = rewrite_response
        self.answer_response = answer_response

    def with_structured_output(self, schema):
        return StructuredResponder(self, schema)


class FakeRetrievalPipeline:
    def __init__(self, top_chunks):
        self.top_chunks = top_chunks

    def retrieve(self, query: str, *, doc_ids=None):
        return RetrievalResult(
            query=query,
            candidates=self.top_chunks,
            top_chunks=self.top_chunks,
            explanation=make_retrieval_explanation(query),
            debug={
                "dense_hit_count": len(self.top_chunks),
                "top_chunk_ids": [chunk.chunk_id for chunk in self.top_chunks],
            },
        )


def build_test_client(tmp_path: Path) -> tuple[TestClient, FakeRuntime]:
    thread_store = LocalThreadStore(tmp_path / "storage" / "threads")
    agent = FakeAgent()
    runtime = FakeRuntime(
        settings=Settings(
            openrouter_api_key="test-key",
            storage_root=tmp_path / "storage",
        ),
        catalog=FakeCatalog(),
        agent=agent,
        thread_store=thread_store,
        thread_service=ThreadService(thread_store, agent),
    )
    app = create_app(runtime=runtime)
    return TestClient(app), runtime


def build_real_agent_client(tmp_path: Path, monkeypatch, *, top_chunks, rewrite_response, answer_response):
    thread_store = LocalThreadStore(tmp_path / "storage" / "threads")
    settings = Settings(
        openrouter_api_key="test-key",
        storage_root=tmp_path / "storage",
    )
    monkeypatch.setattr(
        "backend.agent.runner.get_chat_model",
        lambda _settings: FakeChatModel(rewrite_response, answer_response),
    )
    agent = AgentRunner(
        AgentDependencies(
            settings=settings,
            catalog=FakeCatalog(),
            retrieval_pipeline=FakeRetrievalPipeline(top_chunks),
            tools=[],
            tool_registry={},
        )
    )
    runtime = FakeRuntime(
        settings=settings,
        catalog=FakeCatalog(),
        agent=agent,
        thread_store=thread_store,
        thread_service=ThreadService(thread_store, agent),
    )
    return TestClient(create_app(runtime=runtime))


def test_api_health_library_and_chat(tmp_path: Path):
    client, _runtime = build_test_client(tmp_path)

    health = client.get("/api/health")
    library = client.get("/api/library")
    chat = client.post("/api/chat", json={"question": "What is this?", "debug": True})

    assert health.status_code == 200
    assert library.status_code == 200
    assert chat.status_code == 200
    assert chat.json()["used_doc_ids"] == ["demo-guideline"]
    assert chat.json()["retrieval_explanation"]["query_used"] == "What is this?"
    assert chat.json()["debug_trace"][0]["step"] == "dense_retrieval"


def test_chat_returns_gateway_timeout_for_openrouter_524_validation_error(tmp_path: Path):
    client, runtime = build_test_client(tmp_path)
    runtime.agent.error = RuntimeError(
        "Response validation failed: body.choices Field required "
        "[type=missing, input_value={'error': {'message': 'Provider timed out', 'code': 524}}, input_type=dict]"
    )

    response = client.post("/api/chat", json={"question": "What is this?"})

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


def test_thread_api_crud_and_append_message(tmp_path: Path):
    client, runtime = build_test_client(tmp_path)

    created = client.post(
        "/api/threads",
        json={"scope": {"doc_ids": ["demo-guideline"]}},
    )
    assert created.status_code == 201
    thread = created.json()
    thread_id = thread["id"]

    listed = client.get("/api/threads")
    fetched = client.get(f"/api/threads/{thread_id}")
    appended = client.post(
        f"/api/threads/{thread_id}/messages",
        json={"message": "What is this guideline about?", "debug": True},
    )

    assert listed.status_code == 200
    assert fetched.status_code == 200
    assert appended.status_code == 200
    assert listed.json()[0]["id"] == thread_id
    assert fetched.json()["doc_ids"] == ["demo-guideline"]

    messages = appended.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[-1]["citations"][0]["chunk_id"] == "demo-guideline::chunk_0000"
    assert messages[-1]["retrieval_explanation"]["query_used"] == "What is this guideline about?"
    assert messages[-1]["debug_trace"][0]["step"] == "dense_retrieval"
    assert appended.json()["title"] == "What is this guideline about?"

    stored_path = tmp_path / "storage" / "threads" / f"{thread_id}.json"
    stored_payload = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored_payload["messages"][-1]["role"] == "assistant"
    assert (
        stored_payload["messages"][-1]["retrieval_explanation"]["final_supporting_chunks"]["items"][0]["chunk_id"]
        == "demo-guideline::chunk_0000"
    )

    deleted = client.delete(f"/api/threads/{thread_id}")
    missing = client.get(f"/api/threads/{thread_id}")

    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert runtime.agent.calls[0]["doc_ids"] == ["demo-guideline"]


def test_thread_append_returns_gateway_timeout_for_openrouter_524_validation_error(tmp_path: Path):
    client, runtime = build_test_client(tmp_path)
    runtime.agent.error = RuntimeError(
        "Response validation failed: body.id Field required "
        "[type=missing, input_value={'error': {'message': 'Provider timed out', 'code': 524}}, input_type=dict]"
    )

    created = client.post("/api/threads", json={})
    thread_id = created.json()["id"]
    response = client.post(
        f"/api/threads/{thread_id}/messages",
        json={"message": "What is this guideline about?"},
    )

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


def test_thread_scope_update_persists_and_passes_prior_turns(tmp_path: Path):
    client, runtime = build_test_client(tmp_path)

    created = client.post("/api/threads", json={})
    thread_id = created.json()["id"]

    updated = client.patch(
        f"/api/threads/{thread_id}",
        json={"title": "Renamed thread", "scope": {"doc_ids": ["demo-guideline"]}},
    )
    first_reply = client.post(
        f"/api/threads/{thread_id}/messages",
        json={"message": "Start with the screening schedule."},
    )
    second_reply = client.post(
        f"/api/threads/{thread_id}/messages",
        json={"message": "What about contraindications?"},
    )

    assert updated.status_code == 200
    assert updated.json()["title"] == "Renamed thread"
    assert updated.json()["doc_ids"] == ["demo-guideline"]
    assert first_reply.status_code == 200
    assert second_reply.status_code == 200

    second_call = runtime.agent.calls[-1]
    assert second_call["doc_ids"] == ["demo-guideline"]
    assert second_call["prior_turns"] == [
        {"role": "user", "content": "Start with the screening schedule."},
        {"role": "assistant", "content": "Answered: Start with the screening schedule."},
    ]


def test_create_app_serves_built_spa_when_present(tmp_path: Path):
    client, runtime = build_test_client(tmp_path)
    frontend_dist = tmp_path / "frontend" / "dist"
    assets_dir = frontend_dist / "assets"
    assets_dir.mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<html><body>spa-shell</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")

    app = create_app(runtime=runtime, frontend_dist=frontend_dist)
    client = TestClient(app)

    root = client.get("/")
    thread_route = client.get("/threads/demo-thread")
    asset = client.get("/assets/app.js")

    assert root.status_code == 200
    assert "spa-shell" in root.text
    assert thread_route.status_code == 200
    assert "spa-shell" in thread_route.text
    assert asset.status_code == 200
    assert "console.log('ok');" in asset.text


def test_api_chat_uses_real_agent_runner_and_conservative_fallback(tmp_path: Path, monkeypatch):
    client = build_real_agent_client(
        tmp_path,
        monkeypatch,
        top_chunks=[],
        rewrite_response=RewrittenQuery(query="dose query"),
        answer_response=AnswerDraft(answer="unused"),
    )

    response = client.post("/api/chat", json={"question": "What is the dose?", "debug": True})

    assert response.status_code == 200
    assert response.json()["answer"] == "I don't know based on the indexed guidelines."
    assert response.json()["citations"] == []
    assert response.json()["retrieval_explanation"]["query_used"] == "What is the dose?"
    assert response.json()["retrieval_explanation"]["refined_question_used"] == "dose query"
    assert response.json()["retrieval_explanation"]["lexical_hits"]["items"] == []
    assert [entry["step"] for entry in response.json()["debug_trace"]] == [
        "user",
        "rewrite_query",
        "dense_retrieval",
        "generate_answer",
    ]


def test_api_chat_returns_citations_from_real_agent_runner(tmp_path: Path, monkeypatch):
    client = build_real_agent_client(
        tmp_path,
        monkeypatch,
        top_chunks=[
            RetrievedChunk(
                doc_id="demo-guideline",
                chunk_id="demo-guideline::chunk_0000",
                chunk_index=0,
                breadcrumbs="Treatment",
                text="Use ibuprofen for mild pain.",
                source_path="demo.md",
            )
        ],
        rewrite_response=RewrittenQuery(query="mild pain query"),
        answer_response=AnswerDraft(
            answer="Use ibuprofen for mild pain.",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )

    response = client.post("/api/chat", json={"question": "What should I use for mild pain?", "debug": True})

    assert response.status_code == 200
    assert response.json()["used_doc_ids"] == ["demo-guideline"]
    assert response.json()["citations"][0]["chunk_id"] == "demo-guideline::chunk_0000"
    assert response.json()["retrieval_explanation"]["query_used"] == "What should I use for mild pain?"
    assert response.json()["retrieval_explanation"]["refined_question_used"] == "mild pain query"
    assert response.json()["retrieval_explanation"]["dense_hits"]["items"][0]["chunk_id"] == (
        "demo-guideline::chunk_0000"
    )
    assert response.json()["retrieval_explanation"]["final_supporting_chunks"]["items"][0]["chunk_id"] == (
        "demo-guideline::chunk_0000"
    )
    assert [entry["step"] for entry in response.json()["debug_trace"]] == [
        "user",
        "rewrite_query",
        "dense_retrieval",
        "generate_answer",
    ]


def test_api_chat_returns_generic_server_error_for_non_timeout_failures(tmp_path: Path):
    client, runtime = build_test_client(tmp_path)
    runtime.agent.error = RuntimeError("Boom")

    response = client.post("/api/chat", json={"question": "What is this?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Boom"


def test_api_chat_clears_citations_for_conservative_fallback_answer(tmp_path: Path, monkeypatch):
    client = build_real_agent_client(
        tmp_path,
        monkeypatch,
        top_chunks=[
            RetrievedChunk(
                doc_id="demo-guideline",
                chunk_id="demo-guideline::chunk_0000",
                chunk_index=0,
                breadcrumbs="Treatment",
                text="Use ibuprofen for mild pain.",
                source_path="demo.md",
            )
        ],
        rewrite_response=RewrittenQuery(query="mild pain query"),
        answer_response=AnswerDraft(
            answer="I don't know based on the indexed guidelines.",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )

    response = client.post("/api/chat", json={"question": "What should I use for mild pain?", "debug": True})

    assert response.status_code == 200
    assert response.json()["answer"] == "I don't know based on the indexed guidelines."
    assert response.json()["citations"] == []
    assert response.json()["used_doc_ids"] == []
    assert response.json()["retrieval_explanation"]["final_supporting_chunks"]["items"] == []


def test_api_chat_normalizes_non_english_fallback_answer(tmp_path: Path, monkeypatch):
    client = build_real_agent_client(
        tmp_path,
        monkeypatch,
        top_chunks=[
            RetrievedChunk(
                doc_id="demo-guideline",
                chunk_id="demo-guideline::chunk_0000",
                chunk_index=0,
                breadcrumbs="Treatment",
                text="Use ibuprofen for mild pain.",
                source_path="demo.md",
            )
        ],
        rewrite_response=RewrittenQuery(query="mild pain query"),
        answer_response=AnswerDraft(
            answer="Ma ei tea, mida soovitada, sest juhendis ei ole selle kohta infot.",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )

    response = client.post("/api/chat", json={"question": "What should I use for mild pain?", "debug": True})

    assert response.status_code == 200
    assert response.json()["answer"] == "I don't know based on the indexed guidelines."
    assert response.json()["citations"] == []
    assert response.json()["used_doc_ids"] == []
    assert response.json()["retrieval_explanation"]["final_supporting_chunks"]["items"] == []
