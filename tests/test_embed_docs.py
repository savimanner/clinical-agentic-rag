from pathlib import Path
import sys

from chromadb.errors import InvalidArgumentError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from embed_docs import is_dimension_mismatch, replace_document_embeddings


class FakeCollection:
    def __init__(self):
        self.calls = []

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))


def test_replace_document_embeddings_force_deletes_full_document_slice():
    collection = FakeCollection()

    replace_document_embeddings(
        collection,
        doc_id="demo-guideline",
        ids=["demo-guideline::chunk_0000"],
        vectors=[[0.1, 0.2]],
        metadatas=[{"doc_id": "demo-guideline"}],
        documents=["chunk body"],
        force=True,
    )

    assert collection.calls[0] == ("delete", {"where": {"doc_id": "demo-guideline"}})
    assert collection.calls[1][0] == "upsert"


def test_replace_document_embeddings_skips_delete_without_force():
    collection = FakeCollection()

    replace_document_embeddings(
        collection,
        doc_id="demo-guideline",
        ids=["demo-guideline::chunk_0000"],
        vectors=[[0.1, 0.2]],
        metadatas=[{"doc_id": "demo-guideline"}],
        documents=["chunk body"],
        force=False,
    )

    assert [name for name, _ in collection.calls] == ["upsert"]


def test_is_dimension_mismatch_matches_chroma_error_message():
    exc = InvalidArgumentError("Collection expecting embedding with dimension of 3072, got 2048")

    assert is_dimension_mismatch(exc) is True


def test_is_dimension_mismatch_ignores_other_chroma_errors():
    exc = InvalidArgumentError("some other validation error")

    assert is_dimension_mismatch(exc) is False
