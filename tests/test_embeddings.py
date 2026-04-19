from backend.core.embeddings import OpenRouterEmbeddings


class DummyItem:
    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class DummyResponse:
    def __init__(self, data):
        self.data = data


class FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, list(input)))
        if model == "bad-model":
            raise ValueError("No embedding data received")
        return DummyResponse([DummyItem([0.1, 0.2], idx) for idx, _ in enumerate(input)])


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsAPI()


def test_openrouter_embeddings_fall_back_when_primary_model_returns_no_vectors():
    embeddings = OpenRouterEmbeddings(
        api_key="test",
        model="bad-model",
        fallback_model="openai/text-embedding-3-large",
        base_url="https://openrouter.ai/api/v1",
    )
    fake_client = FakeClient()
    embeddings._client = fake_client

    result = embeddings.embed_query("hello")

    assert result == [0.1, 0.2]
    assert fake_client.embeddings.calls == [
        ("bad-model", ["hello"]),
        ("openai/text-embedding-3-large", ["hello"]),
    ]
