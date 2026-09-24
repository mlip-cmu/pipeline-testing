from datetime import date

import pytest

from compound.agent import react_agent
from compound.tools import sample_workspace
from fakes import RuleBasedAgentLLM, ScriptedLLM

TODAY = date(2026, 9, 23)


def test_agent_terminates():
    """Agent should reach a final answer within step budget."""
    result = react_agent("What meetings do I have tomorrow?", max_steps=10, llm=RuleBasedAgentLLM(), today=TODAY)
    assert result.terminated
    assert result.steps_taken <= 10
    assert "Project sync with Alice" in result.answer


def test_agent_stops_at_step_budget():
    llm = ScriptedLLM(*['ACTION: get_email {"start": "2026-09-23", "end": "2026-09-23"}'] * 4)
    result = react_agent("Summarize my email", max_steps=4, llm=llm, today=TODAY)
    assert not result.terminated
    assert result.steps_taken == 4
    assert len(llm.prompts) == 4


def test_agent_handles_tool_failure():
    """Agent should gracefully handle a tool that errors out."""
    def failing_calendar(start, end):
        raise ConnectionError("Calendar API unavailable")

    llm = RuleBasedAgentLLM()
    result = react_agent("What meetings do I have tomorrow?", tools={"get_calendar": failing_calendar},
                         max_steps=10, llm=llm, today=TODAY)
    assert "unavailable" in result.answer.lower() or result.terminated
    assert "Calendar API unavailable" in llm.prompts[-1]


def test_tool_failure_is_recorded_outside_the_answer():
    """Fail-plausible: the model may turn the error into a fluent answer, so check the harness state."""
    def failing_calendar(start, end):
        raise ConnectionError("Calendar API unavailable")

    llm = ScriptedLLM('ACTION: get_calendar {"start": "2026-09-24", "end": "2026-09-24"}',
                      "FINAL ANSWER: You have no meetings tomorrow.")
    result = react_agent("What meetings do I have tomorrow?", tools={"get_calendar": failing_calendar},
                         llm=llm, today=TODAY)
    assert result.answer == "You have no meetings tomorrow."
    assert result.tool_errors == ["get_calendar: ConnectionError: Calendar API unavailable"]


def test_unknown_tool_is_reported_to_llm():
    llm = ScriptedLLM("ACTION: delete_everything {}", "FINAL ANSWER: done")
    react_agent("Clean up", llm=llm, today=TODAY)
    assert "unknown tool delete_everything" in llm.prompts[1]


def test_unparseable_reply_is_reported_to_llm():
    llm = ScriptedLLM("Let me think about this...", "FINAL ANSWER: done")
    result = react_agent("Hi", llm=llm, today=TODAY)
    assert "could not parse" in llm.prompts[1]
    assert result.steps_taken == 2


def test_invalid_tool_arguments_are_reported_to_llm():
    llm = ScriptedLLM('ACTION: get_calendar {"start": "tomorrow"}', "FINAL ANSWER: done")
    react_agent("Meetings?", llm=llm, today=TODAY)
    assert "get_calendar failed" in llm.prompts[1]


@pytest.mark.parametrize("approved, sent", [(True, 1), (False, 0)])
def test_sending_email_needs_user_approval(approved, sent):
    workspace = sample_workspace(TODAY)
    workspace.confirm_send = lambda email: approved
    llm = ScriptedLLM('ACTION: send_email {"to": "alice@example.com", "subject": "Sync", "body": "3pm works."}',
                      "FINAL ANSWER: ok")
    react_agent("Tell Alice 3pm works", tools=workspace.tools(), llm=llm, today=TODAY)
    assert len(workspace.outbox) == sent

