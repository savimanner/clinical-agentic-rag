from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.agent.graph import AgentDependencies, build_agent_graph


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
    ) -> dict:
        initial_state = {
            "question": question,
            "doc_ids": doc_ids or None,
            "working_doc_ids": doc_ids or None,
            "debug": debug,
            "messages": [HumanMessage(content=question)],
            "iteration_count": 0,
            "tool_call_count": 0,
            "retrieval_attempt_count": 0,
            "retrieval_attempted": False,
            "last_tool_names": [],
            "trace": [{"step": "user", "question": question, "doc_ids": doc_ids or []}],
        }
        result = self._graph.invoke(initial_state, config={"recursion_limit": 32})
        return result["final_payload"]
