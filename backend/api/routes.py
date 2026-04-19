from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.api.schemas import (
    AppendMessageRequest,
    ChatRequest,
    ChatResponse,
    CreateThreadRequest,
    HealthResponse,
    ThreadDetail,
    ThreadSummary,
    UpdateThreadRequest,
)


router = APIRouter()


def _validate_doc_ids(runtime, doc_ids: list[str] | None) -> list[str]:
    validated: list[str] = []
    for doc_id in doc_ids or []:
        if runtime.catalog.get_document(doc_id) is None:
            raise HTTPException(status_code=422, detail=f"Unknown doc_id: {doc_id}")
        validated.append(doc_id)
    return validated


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


@router.get("/api/threads", response_model=list[ThreadSummary])
def list_threads(request: Request) -> list[ThreadSummary]:
    runtime = request.app.state.runtime
    return runtime.thread_service.list_threads()


@router.post("/api/threads", response_model=ThreadDetail, status_code=status.HTTP_201_CREATED)
def create_thread(request: Request, payload: CreateThreadRequest | None = None) -> ThreadDetail:
    runtime = request.app.state.runtime
    body = payload or CreateThreadRequest()
    doc_ids = _validate_doc_ids(runtime, body.resolved_doc_ids())
    return runtime.thread_service.create_thread(title=body.title, doc_ids=doc_ids)


@router.get("/api/threads/{thread_id}", response_model=ThreadDetail)
def get_thread(thread_id: str, request: Request) -> ThreadDetail:
    runtime = request.app.state.runtime
    thread = runtime.thread_service.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return thread


@router.patch("/api/threads/{thread_id}", response_model=ThreadDetail)
def update_thread(
    thread_id: str,
    payload: UpdateThreadRequest,
    request: Request,
) -> ThreadDetail:
    runtime = request.app.state.runtime
    doc_ids = None
    if payload.doc_ids is not None or payload.scope is not None:
        doc_ids = _validate_doc_ids(runtime, payload.resolved_doc_ids())
    thread = runtime.thread_service.update_thread(
        thread_id,
        title=payload.title,
        title_set="title" in payload.model_fields_set,
        doc_ids=doc_ids,
        doc_ids_set="doc_ids" in payload.model_fields_set or "scope" in payload.model_fields_set,
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return thread


@router.delete("/api/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(thread_id: str, request: Request) -> Response:
    runtime = request.app.state.runtime
    deleted = runtime.thread_service.delete_thread(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/api/threads/{thread_id}/messages", response_model=ThreadDetail)
def append_thread_message(
    thread_id: str,
    payload: AppendMessageRequest,
    request: Request,
) -> ThreadDetail:
    runtime = request.app.state.runtime
    if not runtime.settings.openrouter_api_key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured.")
    try:
        updated = runtime.thread_service.append_message(
            thread_id,
            content=payload.content,
            debug=payload.debug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced in manual runs
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return updated


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
