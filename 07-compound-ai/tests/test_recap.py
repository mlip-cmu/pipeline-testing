from pathlib import Path

import pytest

from compound.context import InvalidModelReply
from compound.recap import SAMPLE_MESSAGES, Message, summarize_unread
from compound.replay import MissingRecording, ReplayLLM
from fakes import ScriptedLLM

MESSAGES = SAMPLE_MESSAGES
RECORDINGS = Path(__file__).parents[1] / "recordings"


def test_recap_with_recorded_reply():
    llm = ReplayLLM(RECORDINGS / "recap.json")
    recap = summarize_unread(MESSAGES, llm=llm)
    assert recap.message_ids == [3, 7]


def test_changed_prompt_has_no_recording():
    llm = ReplayLLM(RECORDINGS / "recap.json")
    with pytest.raises(MissingRecording):
        summarize_unread(MESSAGES, llm=llm, user="Ann")


def test_prompt_contains_only_unread_messages():
    llm = ScriptedLLM('{"summary": "ok", "message_ids": []}')
    summarize_unread(MESSAGES, llm=llm)
    assert "[3] carol" in llm.prompts[0]
    assert "[1] alice" not in llm.prompts[0]


def test_no_model_call_without_unread_messages():
    llm = ScriptedLLM()
    recap = summarize_unread([Message(1, "alice", "hi", read=True)], llm=llm)
    assert recap.message_ids == []
    assert llm.prompts == []


def test_recap_recovers_from_invalid_json():
    llm = ScriptedLLM('{"summary": "Standup moved to',  # cut off
                      '{"summary": "Standup moved to 10am"}')
    recap = summarize_unread(MESSAGES, llm=llm)
    assert recap.summary == "Standup moved to 10am"
    assert len(llm.prompts) == 2                        # retried once
    assert "invalid JSON" in llm.prompts[-1]            # error sent back


def test_recap_fails_loudly_after_repeated_invalid_json():
    llm = ScriptedLLM("Sure! Here is your recap:", "{}")
    with pytest.raises(InvalidModelReply):
        summarize_unread(MESSAGES, llm=llm)


def test_recap_ignores_ids_of_other_messages():
    llm = ScriptedLLM('{"summary": "...", "message_ids": [3, 1, 99]}')
    assert summarize_unread(MESSAGES, llm=llm).message_ids == [3]
