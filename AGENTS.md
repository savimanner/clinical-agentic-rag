# AGENTS.md

This repository contains a local-first agentic RAG application built with FastAPI, LangGraph, LangChain, OpenRouter, and Chroma.

## Default expectations

- Speak to the user in English unless they ask otherwise.
- Prefer small, targeted changes over broad rewrites.
- Preserve the current structure:
  - `backend/` contains runtime application code
  - `scripts/` contains ingestion and indexing entrypoints
  - `data/` contains local staged document artifacts
  - `storage/chroma/` contains the persistent vector database

## Runtime entrypoints

- FastAPI app entrypoint: `backend.main:app`
- Start locally with:
  - `./.venv/bin/uvicorn backend.main:app --reload`
- Health check:
  - `http://127.0.0.1:8000/api/health`

## Frontend runtime

- Frontend app lives in `frontend/`
- Install frontend dependencies with:
  - `cd frontend && npm install`
- Start the frontend dev server with:
  - `cd frontend && npm run dev`
- Frontend dev URL:
  - `http://127.0.0.1:5173`
- The Vite dev server proxies `/api` to:
  - `http://127.0.0.1:8000`
- To build the frontend for FastAPI to serve from `frontend/dist`, run:
  - `cd frontend && npm run build`

## Ingestion workflow

Run these from the repository root as needed:

1. Convert PDFs to markdown:
   - `./.venv/bin/python scripts/pdf_to_markdown.py`
2. Normalize markdown:
   - `./.venv/bin/python scripts/normalize_markdown.py`
3. Rebuild chunk files:
   - `./.venv/bin/python scripts/chunk_markdown.py --force`
4. Embed chunks into Chroma:
   - `./.venv/bin/python scripts/embed_docs.py`
   - Use `./.venv/bin/python scripts/embed_docs.py --force` after chunk changes or embedding-model changes.
   - If the embedding model dimension differs from the existing Chroma collection, the force rebuild path will recreate the collection before re-indexing.

## Data and indexing assumptions

- Chunks in `data/*/30_chunks/*.jsonl` are the canonical generated chunk artifacts.
- Embeddings must be persisted in `storage/chroma/`; they should not depend on a browser session.
- The application should read from the persisted Chroma store at startup.
- If chunk files change, re-run `scripts/embed_docs.py`.
- If `OPENROUTER_EMBEDDING_MODEL` changes, re-run `scripts/embed_docs.py --force` so the persisted Chroma index matches the embedding dimension.

## Environment variables

Expected in `.env`:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_EMBEDDING_MODEL`
- Optional:
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_PROJECT`

Recommended embedding model:

- `OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-large`
- The repo has also been verified with `OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free`.

## Agent/runtime notes

- The retrieval loop is implemented in `backend/agent/graph.py`.
- Metadata tools like `search_library` and `get_document_outline` should lead into retrieval, not terminate the loop early.
- If changing control flow, verify both:
  - automated tests in `tests/`
  - a live question through `POST /api/chat`

## Testing

- Run all tests:
  - `./.venv/bin/pytest -q`
- Before finishing substantial code changes, at minimum run:
  - relevant targeted tests
  - `./.venv/bin/python -m compileall backend scripts tests`

## Git

- Do not commit `data/`, `storage/`, `.env`, or local caches unless explicitly asked.
- Keep commits focused and descriptive.
