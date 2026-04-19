from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from backend.api.frontend import frontend_dist_dir, serve_frontend_index
from backend.api.routes import router
from backend.core.runtime import build_runtime


LOCAL_VITE_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _default_frontend_dist() -> Path:
    return frontend_dist_dir()


def _configure_frontend_routes(app: FastAPI, frontend_dist: Path) -> None:
    index_path = frontend_dist / "index.html"
    if not index_path.exists():
        @app.get("/", include_in_schema=False)
        def root_fallback():
            return serve_frontend_index()

        @app.get("/threads/{thread_id}", include_in_schema=False)
        def thread_fallback(thread_id: str):
            return serve_frontend_index()

        return

    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/threads/{thread_id}", include_in_schema=False)
    def serve_frontend_thread(thread_id: str) -> FileResponse:
        return FileResponse(index_path)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend_app(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found.")
        candidate = frontend_dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_path)


def create_app(runtime=None, *, frontend_dist: Path | None = None) -> FastAPI:
    runtime = runtime or build_runtime()
    app = FastAPI(title=runtime.settings.app_name)
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=LOCAL_VITE_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    _configure_frontend_routes(app, frontend_dist or _default_frontend_dist())
    return app
