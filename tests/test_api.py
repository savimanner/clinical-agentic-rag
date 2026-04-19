from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from backend.agent.graph import AgentDependencies
from backend.agent.runner import AgentRunner
from backend.agent.schemas import AnswerDraft, EvidenceGrade
from backend.api.app import create_app
from backend.content.catalog import DocumentSummary
from backend.core.settings import Settings
from backend.rag.models import RetrievedChunk
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


@dataclass
class FakeAgent:
    calls: list[dict] = field(default_factory=list)

    def answer_question(self, question: str, *, doc_ids=None, debug=False, prior_turns=None):
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
            "debug_trace": [{"step": "planner"}] if debug else None,
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
        if self.schema is EvidenceGrade:
            return self.parent.grade_response
        return self.parent.answer_response


class FakeChatModel:
    def __init__(self, grade_response, answer_response):
        self.grade_response = grade_response
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
            debug={
                "lexical_hit_count": len(self.top_chunks),
                "dense_hit_count": len(self.top_chunks),
                "candidate_count": len(self.top_chunks),
                "top_chunk_ids": [chunk.chunk_id for chunk in self.top_chunks],
                "rerank_reasoning": "fake",
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


def build_real_agent_client(tmp_path: Path, monkeypatch, *, top_chunks, grade_response, answer_response):
    thread_store = LocalThreadStore(tmp_path / "storage" / "threads")
    settings = Settings(
        openrouter_api_key="test-key",
        storage_root=tmp_path / "storage",
    )
    monkeypatch.setattr(
        "backend.agent.graph.get_chat_model",
        lambda _settings: FakeChatModel(grade_response, answer_response),
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
    assert chat.json()["debug_trace"][0]["step"] == "planner"


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
    assert messages[-1]["debug_trace"][0]["step"] == "planner"
    assert appended.json()["title"] == "What is this guideline about?"

    stored_path = tmp_path / "storage" / "threads" / f"{thread_id}.json"
    stored_payload = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored_payload["messages"][-1]["role"] == "assistant"

    deleted = client.delete(f"/api/threads/{thread_id}")
    missing = client.get(f"/api/threads/{thread_id}")

    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert runtime.agent.calls[0]["doc_ids"] == ["demo-guideline"]


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
        grade_response=EvidenceGrade(sufficient=False, reasoning="No evidence."),
        answer_response=AnswerDraft(answer="unused"),
    )

    response = client.post("/api/chat", json={"question": "What is the dose?", "debug": True})

    assert response.status_code == 200
    assert response.json()["answer"] == "I don't know based on the indexed guidelines."
    assert response.json()["citations"] == []


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
        grade_response=EvidenceGrade(sufficient=True, reasoning="Enough evidence."),
        answer_response=AnswerDraft(
            answer="Use ibuprofen for mild pain.",
            cited_chunk_ids=["demo-guideline::chunk_0000"],
        ),
    )

    response = client.post("/api/chat", json={"question": "What should I use for mild pain?"})

    assert response.status_code == 200
    assert response.json()["used_doc_ids"] == ["demo-guideline"]
    assert response.json()["citations"][0]["chunk_id"] == "demo-guideline::chunk_0000"
