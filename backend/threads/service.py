from __future__ import annotations

from backend.api.schemas import ThreadDetail, ThreadSummary
from backend.threads.store import LocalThreadStore


class ThreadService:
    def __init__(self, store: LocalThreadStore, agent) -> None:
        self.store = store
        self.agent = agent

    def list_threads(self) -> list[ThreadSummary]:
        return self.store.list_threads()

    def create_thread(self, *, title: str | None = None, doc_ids: list[str] | None = None) -> ThreadDetail:
        return self.store.create_thread(title=title, doc_ids=doc_ids)

    def get_thread(self, thread_id: str) -> ThreadDetail | None:
        return self.store.get_thread(thread_id)

    def update_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        title_set: bool = False,
        doc_ids: list[str] | None = None,
        doc_ids_set: bool = False,
    ) -> ThreadDetail | None:
        next_title = title if title_set else None
        next_doc_ids = doc_ids if doc_ids_set else None
        return self.store.update_thread(thread_id, title=next_title, doc_ids=next_doc_ids)

    def delete_thread(self, thread_id: str) -> bool:
        return self.store.delete_thread(thread_id)

    def append_message(self, thread_id: str, *, content: str, debug: bool = False) -> ThreadDetail | None:
        thread = self.store.get_thread(thread_id)
        if thread is None:
            return None

        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("Message content cannot be empty.")

        history = [{"role": message.role, "content": message.content} for message in thread.messages]
        assistant_payload = self.agent.answer_question(
            normalized_content,
            doc_ids=thread.doc_ids or None,
            debug=debug,
            prior_turns=history,
        )
        return self.store.append_exchange(
            thread_id,
            user_message=normalized_content,
            assistant_payload=assistant_payload,
        )
