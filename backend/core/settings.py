from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Local Agentic RAG"
    app_environment: str = "local"

    data_root: Path = Field(default=Path("data"))
    storage_root: Path = Field(default=Path("storage"))
    chroma_persist_directory: Path = Field(default=Path("storage/chroma"))
    chroma_collection_name: str = "guidelines"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    openrouter_referer: str = "http://localhost"
    openrouter_app_title: str = "fastapi-learning-rag"

    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_project: str = "fastapi-learning-rag"
    langsmith_tracing: bool = True

    retrieval_k: int = 5
    retrieval_fetch_k: int = 12
    retrieval_lexical_k: int = 24
    retrieval_candidate_k: int = 32
    retrieval_final_k: int = 8
    chunk_target_chars: int = 1000
    chunk_overlap_chars: int = 120
    chunk_hard_max_chars: int = 1600

    agent_history_turn_limit: int = 4
    debug_context_limit: int = 8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chroma_directory(self) -> Path:
        return self.chroma_persist_directory

    @property
    def threads_directory(self) -> Path:
        return self.storage_root / "threads"

    @property
    def index_exists(self) -> bool:
        return self.chroma_directory.exists() and any(self.chroma_directory.iterdir())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
