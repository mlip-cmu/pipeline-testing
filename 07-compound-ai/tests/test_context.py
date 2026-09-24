"""The bugs from the slide "Subtle Bugs in Context Assembly Code" and tests that catch them."""

import pytest

from compound import context, context_buggy
from compound.context import count_tokens, message

SYSTEM = message("You summarize chat messages.", role="system")
BUGGY = pytest.mark.xfail(strict=True, reason="bug from the slide")


def versions(name, *buggy_names):
    return [pytest.param(getattr(context_buggy, n), marks=BUGGY, id=f"buggy.{n}") for n in buggy_names or [name]] + \
           [pytest.param(getattr(context, name), id=f"fixed.{name}")]


@pytest.mark.parametrize("build_context", versions("build_context", "build_context", "build_context_tail"))
def test_truncation_keeps_system_prompt_and_newest(build_context):
    history = [message(f"m{i}") for i in range(100)]
    messages = build_context(SYSTEM, history, token_budget=500)
    assert messages[0] == SYSTEM
    assert messages[-1] == history[-1]
    assert count_tokens(messages) <= 500


@pytest.mark.parametrize("render", versions("render"))
def test_missing_template_variable_is_an_error(render):
    with pytest.raises(Exception, match="user"):
        render("Recap for {{ user }}: {{ msgs }}", msgs="[3] carol: Standup moves to 10am.")


@pytest.mark.parametrize("fits", versions("fits"))
def test_budget_is_checked_in_tokens(fits):
    assert fits("word " * 50, token_budget=100)
    assert not fits("word " * 1000, token_budget=100)


@pytest.mark.parametrize("parse_recap", versions("parse_recap"))
def test_cut_off_reply_is_not_turned_into_empty_summary(parse_recap):
    with pytest.raises(ValueError):
        parse_recap('{"summary": "Standup moved to')


@pytest.mark.parametrize("call_tool", versions("call_tool"))
def test_tool_failure_is_visible_to_the_model(call_tool):
    def calendar(day):
        raise ConnectionError("Calendar API unavailable")
    assert "Calendar API unavailable" in call_tool(calendar, {"day": "2026-09-25"})
