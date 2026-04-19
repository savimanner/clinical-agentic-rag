from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from backend.agent.graph import AgentDependencies
from backend.agent.runner import AgentRunner
from backend.content.catalog import ContentCatalog
from backend.core.settings import Settings, get_settings
from backend.core.tracing import configure_langsmith
from backend.rag.sources import LocalCorpusSource
from backend.rag.tools import build_rag_tools


@dataclass
class AppRuntime:
    settings: Settings
    catalog: ContentCatalog
    source: LocalCorpusSource
    agent: AgentRunner


def build_runtime(settings: Settings | None = None) -> AppRuntime:
    settings = settings or get_settings()
    configure_langsmith(settings)
    catalog = ContentCatalog(settings.data_root)
    source = LocalCorpusSource(settings, catalog)
    tools, registry = build_rag_tools(source)
    agent = AgentRunner(
        AgentDependencies(
            settings=settings,
            catalog=catalog,
            tools=tools,
            tool_registry=registry,
        )
    )
    return AppRuntime(settings=settings, catalog=catalog, source=source, agent=agent)


@lru_cache(maxsize=1)
def get_runtime() -> AppRuntime:
    return build_runtime()
