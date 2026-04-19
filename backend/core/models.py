from __future__ import annotations

import os

from langchain.chat_models import init_chat_model

from backend.core.settings import Settings


class MissingModelConfigurationError(RuntimeError):
    """Raised when required API configuration is missing."""


def get_chat_model(settings: Settings):
    if not settings.openrouter_api_key:
        raise MissingModelConfigurationError(
            "OPENROUTER_API_KEY is not configured. Set it in your environment or .env."
        )

    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
    return init_chat_model(
        settings.openrouter_model,
        model_provider="openrouter",
        temperature=0,
    )
