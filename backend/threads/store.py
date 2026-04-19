from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.api.schemas import ThreadDetail, ThreadMessage, ThreadSummary
from backend.rag.models import Citation

DEFAULT_THREAD_TITLE = "New thread"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _dedupe_doc_ids(doc_ids: list[str] | None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for doc_id in doc_ids or []:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc_id)
    return deduped


def _title_from_message(message: str) -> str:
    collapsed = " ".join(message.split()).strip()
    if not collapsed:
        return DEFAULT_THREAD_TITLE
    if len(collapsed) <= 72:
        return collapsed
    return f"{collapsed[:69].rstrip()}..."


def _last_message_preview(messages: list[ThreadMessage]) -> str | None:
    if not messages:
        return None
    collapsed = " ".join(messages[-1].content.split()).strip()
    if len(collapsed) <= 96:
        return collapsed
    return f"{collapsed[:93].rstrip()}..."


class StoredThread(BaseModel):
    id: str
    title: str = DEFAULT_THREAD_TITLE
    created_at: datetime
    updated_at: datetime
    doc_ids: list[str] = Field(default_factory=list)
    messages: list[ThreadMessage] = Field(default_factory=list)

    def to_summary(self) -> ThreadSummary:
        return ThreadSummary(
            id=self.id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            doc_ids=list(self.doc_ids),
            message_count=len(self.messages),
            last_message_preview=_last_message_preview(self.messages),
        )

    def to_detail(self) -> ThreadDetail:
        return ThreadDetail(
            id=self.id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            doc_ids=list(self.doc_ids),
            message_count=len(self.messages),
            last_message_preview=_last_message_preview(self.messages),
            messages=list(self.messages),
        )


class LocalThreadStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_threads(self) -> list[ThreadSummary]:
        threads = [thread.to_summary() for thread in self._load_all_threads()]
        return sorted(threads, key=lambda thread: thread.updated_at, reverse=True)

    def create_thread(self, *, title: str | None = None, doc_ids: list[str] | None = None) -> ThreadDetail:
        now = _utcnow()
        thread = StoredThread(
            id=uuid4().hex,
            title=(title or DEFAULT_THREAD_TITLE).strip() or DEFAULT_THREAD_TITLE,
            created_at=now,
            updated_at=now,
            doc_ids=_dedupe_doc_ids(doc_ids),
        )
        self._write_thread(thread)
        return thread.to_detail()

    def get_thread(self, thread_id: str) -> ThreadDetail | None:
        thread = self._load_thread(thread_id)
        return None if thread is None else thread.to_detail()

    def update_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> ThreadDetail | None:
        thread = self._load_thread(thread_id)
        if thread is None:
            return None
        if title is not None:
            thread.title = title.strip() or DEFAULT_THREAD_TITLE
        if doc_ids is not None:
            thread.doc_ids = _dedupe_doc_ids(doc_ids)
        thread.updated_at = _utcnow()
        self._write_thread(thread)
        return thread.to_detail()

    def delete_thread(self, thread_id: str) -> bool:
        path = self._thread_path(thread_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def append_exchange(
        self,
        thread_id: str,
        *,
        user_message: str,
        assistant_payload: dict,
    ) -> ThreadDetail | None:
        thread = self._load_thread(thread_id)
        if thread is None:
            return None

        user_created_at = _utcnow()
        assistant_created_at = _utcnow()
        user_entry = ThreadMessage(
            id=uuid4().hex,
            role="user",
            content=user_message,
            created_at=user_created_at,
        )
        assistant_entry = ThreadMessage(
            id=uuid4().hex,
            role="assistant",
            content=str(assistant_payload.get("answer", "")),
            created_at=assistant_created_at,
            citations=[
                Citation.model_validate(citation)
                for citation in assistant_payload.get("citations", [])
            ],
            used_doc_ids=_dedupe_doc_ids(assistant_payload.get("used_doc_ids", [])),
            debug_trace=assistant_payload.get("debug_trace"),
        )

        if not thread.messages and thread.title == DEFAULT_THREAD_TITLE:
            thread.title = _title_from_message(user_message)

        thread.messages.extend([user_entry, assistant_entry])
        thread.updated_at = assistant_created_at
        self._write_thread(thread)
        return thread.to_detail()

    def _load_all_threads(self) -> list[StoredThread]:
        threads: list[StoredThread] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                threads.append(StoredThread.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return threads

    def _load_thread(self, thread_id: str) -> StoredThread | None:
        path = self._thread_path(thread_id)
        if not path.exists():
            return None
        return StoredThread.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_thread(self, thread: StoredThread) -> None:
        path = self._thread_path(thread.id)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(thread.model_dump_json(indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _thread_path(self, thread_id: str) -> Path:
        return self.root / f"{thread_id}.json"
