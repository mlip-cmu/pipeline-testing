import time

import pytest

from compound.llm import LiteLLM
from compound.recap import SAMPLE_MESSAGES, summarize_unread


def test_litellm_wrapper_with_mocked_provider_response():
    llm = LiteLLM(model="openai/gpt-4o-mini", mock_response="Hello from a mock")
    assert llm("Say hello") == "Hello from a mock"


@pytest.mark.llm
def test_recap_smoke_test_with_real_model():
    start = time.perf_counter()
    recap = summarize_unread(SAMPLE_MESSAGES, LiteLLM())
    assert recap.summary
    assert set(recap.message_ids) <= {3, 4, 5, 7, 8}
    assert time.perf_counter() - start < 60
