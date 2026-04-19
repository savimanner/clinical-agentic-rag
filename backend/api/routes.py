from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.api.ui import INDEX_HTML


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def root() -> str:
    return INDEX_HTML


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    runtime = request.app.state.runtime
    documents = runtime.catalog.list_documents()
    indexed_documents = sum(1 for document in documents if document.indexed)
    return HealthResponse(
        status="ok",
        openrouter_configured=bool(runtime.settings.openrouter_api_key),
        index_exists=runtime.settings.index_exists,
        documents=len(documents),
        indexed_documents=indexed_documents,
    )


@router.get("/api/library")
def library(request: Request):
    runtime = request.app.state.runtime
    return runtime.catalog.list_documents()


@router.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    runtime = request.app.state.runtime
    if not runtime.settings.openrouter_api_key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured.")
    try:
        result = runtime.agent.answer_question(
            payload.question,
            doc_ids=payload.doc_ids,
            debug=payload.debug,
        )
    except Exception as exc:  # pragma: no cover - surfaced in manual runs
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse.model_validate(result)
