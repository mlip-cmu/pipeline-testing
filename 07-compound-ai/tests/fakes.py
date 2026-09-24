"""Test doubles for the LLM."""

import json
import re
from datetime import date, timedelta


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)   # observability: what did the model see?
        return self.replies.pop(0)    # controllability: what does it answer?


class FlakyModel:
    def __init__(self, failures):
        self.failures, self.calls = list(failures), 0

    def __call__(self, prompt):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


class FakeOutbox:
    """Sends the email, but can lose the first acknowledgment. Deduplicates by idempotency key, like a real API."""

    def __init__(self, fail_first_ack=False):
        self.sent, self.keys, self.fail_next_ack = [], set(), fail_first_ack

    def send(self, email, idempotency_key):
        if idempotency_key not in self.keys:
            self.keys.add(idempotency_key)
            self.sent.append(email)
        if self.fail_next_ack:
            self.fail_next_ack = False
            raise TimeoutError("acknowledgment timed out")
        return f"sent-{len(self.sent)}"


class RuleBasedAgentLLM:
    """Behaves like a reasonable model: looks up the calendar once, then answers from the observation."""

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        observations = re.findall(r"^Observation: (.*)$", prompt, re.M)
        if not observations:
            today = date.fromisoformat(re.search(r"Today is (\S+)\.", prompt).group(1))
            tomorrow = (today + timedelta(days=1)).isoformat()
            return f'ACTION: get_calendar {json.dumps({"start": tomorrow, "end": tomorrow})}'
        if observations[-1].startswith("Error"):
            return "FINAL ANSWER: Sorry, your calendar is unavailable right now."
        titles = [event["title"] for event in json.loads(observations[-1])]
        return f"FINAL ANSWER: Tomorrow you have: {', '.join(titles) or 'no meetings'}."
