from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from backend.agent.graph import AgentDependencies
from backend.agent.runner import AgentRunner
from backend.content.catalog import ContentCatalog
from backend.core.settings import Settings, get_settings
from backend.core.tracing import configure_langsmith
from backend.rag.retrieval import HybridRetrievalPipeline
from backend.rag.sources import LocalCorpusSource
from backend.rag.tools import build_rag_tools
from backend.threads import LocalThreadStore, ThreadService


@dataclass
class AppRuntime:
    settings: Settings
    catalog: ContentCatalog
    source: LocalCorpusSource
    agent: AgentRunner
    thread_store: LocalThreadStore
    thread_service: ThreadService


def build_runtime(settings: Settings | None = None) -> AppRuntime:
    settings = settings or get_settings()
    configure_langsmith(settings)
    catalog = ContentCatalog(settings.data_root)
    source = LocalCorpusSource(settings, catalog)
    retrieval_pipeline = HybridRetrievalPipeline(settings, source)
    tools, registry = build_rag_tools(source)
    agent = AgentRunner(
        AgentDependencies(
            settings=settings,
            catalog=catalog,
            retrieval_pipeline=retrieval_pipeline,
            tools=tools,
            tool_registry=registry,
        )
    )
    thread_store = LocalThreadStore(settings.threads_directory)
    thread_service = ThreadService(thread_store, agent)
    return AppRuntime(
        settings=settings,
        catalog=catalog,
        source=source,
        agent=agent,
        thread_store=thread_store,
        thread_service=thread_service,
    )


@lru_cache(maxsize=1)
def get_runtime() -> AppRuntime:
    return build_runtime()
