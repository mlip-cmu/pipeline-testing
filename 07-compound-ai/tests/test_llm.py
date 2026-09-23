import pytest

from compound.agent import react_agent
from compound.llm import LiteLLM
from compound.rag import TfidfRetriever, answer_question, load_documents


def test_litellm_wrapper_with_mocked_provider_response():
    llm = LiteLLM(model="openai/gpt-4o-mini", mock_response="Hello from a mock")
    assert llm("Say hello") == "Hello from a mock"


@pytest.mark.llm
def test_agent_terminates_with_real_llm():
    result = react_agent("What meetings do I have tomorrow?", max_steps=10)
    assert result.terminated
    assert result.steps_taken <= 10


@pytest.mark.llm
def test_agent_handles_tool_failure_with_real_llm():
    def failing_calendar(start, end):
        raise ConnectionError("Calendar API unavailable")

    result = react_agent("What meetings do I have tomorrow?", tools={"get_calendar": failing_calendar}, max_steps=10)
    assert "unavailable" in result.answer.lower() or result.terminated


@pytest.mark.llm
def test_rag_answer_uses_documents():
    answer = answer_question("How much is delivery for a $15 order?", TfidfRetriever(load_documents()), LiteLLM())
    assert "2.99" in answer.text
