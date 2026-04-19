from backend.core.runtime import get_runtime


def answer_question(question: str) -> str:
    runtime = get_runtime()
    result = runtime.agent.answer_question(question)
    return result["answer"]
