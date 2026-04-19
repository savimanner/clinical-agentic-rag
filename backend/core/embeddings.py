from __future__ import annotations

from collections.abc import Iterable

from openai import OpenAI
from langchain_core.embeddings import Embeddings


class OpenRouterEmbeddings(Embeddings):
    """LangChain embeddings adapter backed by OpenRouter's embeddings API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        fallback_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        headers: dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if app_title:
            headers["X-Title"] = app_title
        self._client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)

    def _request_embeddings(self, model: str, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def _embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        batch = list(texts)
        try:
            return self._request_embeddings(self.model, batch)
        except ValueError as exc:
            if "No embedding data received" not in str(exc) or self.model == self.fallback_model:
                raise
            return self._request_embeddings(self.fallback_model, batch)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = 64
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]
