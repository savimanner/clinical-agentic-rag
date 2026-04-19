from dataclasses import dataclass

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.content.catalog import DocumentSummary
from backend.core.settings import Settings


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


class FakeAgent:
    def answer_question(self, question: str, *, doc_ids=None, debug=False):
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


def test_api_endpoints():
    runtime = FakeRuntime(
        settings=Settings(openrouter_api_key="test-key"),
        catalog=FakeCatalog(),
        agent=FakeAgent(),
    )
    app = create_app(runtime=runtime)
    client = TestClient(app)

    health = client.get("/api/health")
    library = client.get("/api/library")
    chat = client.post("/api/chat", json={"question": "What is this?", "debug": True})

    assert health.status_code == 200
    assert library.status_code == 200
    assert chat.status_code == 200
    assert chat.json()["used_doc_ids"] == ["demo-guideline"]
    assert chat.json()["debug_trace"][0]["step"] == "planner"
