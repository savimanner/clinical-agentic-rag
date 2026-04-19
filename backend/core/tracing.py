from __future__ import annotations

import os

from backend.core.settings import Settings


def configure_langsmith(settings: Settings) -> None:
    tracing_enabled = bool(settings.langsmith_tracing and settings.langsmith_api_key)
    os.environ["LANGSMITH_TRACING"] = "true" if tracing_enabled else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    else:
        os.environ.pop("LANGSMITH_API_KEY", None)
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    else:
        os.environ.pop("LANGSMITH_ENDPOINT", None)
