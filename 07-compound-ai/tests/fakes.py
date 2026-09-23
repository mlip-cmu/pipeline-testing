"""Test doubles for the LLM."""

import json
import re
from datetime import date, timedelta


class ScriptedLLM:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


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
