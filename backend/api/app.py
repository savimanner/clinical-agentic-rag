from __future__ import annotations

from fastapi import FastAPI

from backend.api.routes import router
from backend.core.runtime import build_runtime


def create_app(runtime=None) -> FastAPI:
    runtime = runtime or build_runtime()
    app = FastAPI(title=runtime.settings.app_name)
    app.state.runtime = runtime
    app.include_router(router)
    return app
