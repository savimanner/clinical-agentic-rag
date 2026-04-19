from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from backend.agent.graph import AgentDependencies, build_agent_graph


class ConversationTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentRunner:
    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps
        self._graph = build_agent_graph(deps)

    def answer_question(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        debug: bool = False,
        prior_turns: list[ConversationTurn] | None = None,
    ) -> dict:
        bounded_turns = list(prior_turns or [])
        max_history_messages = max(self.deps.settings.agent_history_turn_limit * 2, 0)
        if max_history_messages:
            bounded_turns = bounded_turns[-max_history_messages:]
        else:
            bounded_turns = []

        history_messages = []
        for turn in bounded_turns:
            if turn["role"] == "assistant":
                history_messages.append(AIMessage(content=turn["content"]))
            else:
                history_messages.append(HumanMessage(content=turn["content"]))

        initial_state = {
            "question": question,
            "doc_ids": doc_ids or None,
            "working_doc_ids": doc_ids or None,
            "conversation_history": bounded_turns,
            "debug": debug,
            "messages": [*history_messages, HumanMessage(content=question)],
            "iteration_count": 0,
            "tool_call_count": 0,
            "retrieval_attempt_count": 0,
            "retrieval_attempted": False,
            "last_tool_names": [],
            "trace": [
                {
                    "step": "user",
                    "question": question,
                    "doc_ids": doc_ids or [],
                    "prior_turn_count": len(bounded_turns),
                }
            ],
        }
        result = self._graph.invoke(initial_state, config={"recursion_limit": 32})
        return result["final_payload"]
