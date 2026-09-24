"""Inject model API faults with stubs and test the recovery mechanisms."""

import pytest

from compound.llm import AllModelsFailed, LLMClient, ModelNotFound, RateLimitError
from compound.outbox import Email, send_with_retry
from fakes import FakeOutbox, FlakyModel

email = Email("alice@example.com", "Sync", "3pm works.")


def test_fallback_after_rate_limits():
    primary = FlakyModel([RateLimitError(retry_after=0)] * 3)
    backup = FlakyModel([])
    client = LLMClient([primary, backup], max_attempts=3)
    assert client.complete("hi") == "ok"
    assert primary.calls == 3 and backup.calls == 1
    assert client.fallbacks == 1


def test_retry_obeys_retry_after():
    waits = []
    client = LLMClient([FlakyModel([RateLimitError(retry_after=7), RateLimitError(retry_after=2)])], sleep=waits.append)
    assert client.complete("hi") == "ok"
    assert waits == [7, 2]


def test_retired_model_falls_back_without_retries():
    retired = FlakyModel([ModelNotFound("claude-old")] * 3)
    client = LLMClient([retired, FlakyModel([])], sleep=lambda s: None)
    assert client.complete("hi") == "ok"
    assert retired.calls == 1


def test_error_when_all_models_fail():
    client = LLMClient([FlakyModel([TimeoutError()] * 5)], max_attempts=2, sleep=lambda s: None)
    with pytest.raises(AllModelsFailed):
        client.complete("hi")


def test_no_duplicate_email_after_timeout():
    outbox = FakeOutbox(fail_first_ack=True)  # email sent, ack times out
    send_with_retry(outbox, email, idempotency_key="req-42")
    assert len(outbox.sent) == 1


def test_same_key_is_used_for_all_attempts():
    outbox = FakeOutbox(fail_first_ack=True)
    send_with_retry(outbox, email)
    assert len(outbox.sent) == 1 and len(outbox.keys) == 1
