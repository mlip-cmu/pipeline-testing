"""Recap of unread chat messages, with JSON output that the code checks."""

from dataclasses import dataclass, field

from compound.context import InvalidModelReply, parse_recap
from compound.llm import LLM
from compound.prompts import PromptTemplate


@dataclass
class Message:
    id: int
    sender: str
    text: str
    read: bool = False


@dataclass
class Recap:
    summary: str
    message_ids: list[int] = field(default_factory=list)


SAMPLE_MESSAGES = [
    Message(1, "alice", "Welcome to the team channel!", read=True),
    Message(2, "bob", "Lunch at noon?", read=True),
    Message(3, "carol", "Standup moves to 10am tomorrow."),
    Message(4, "ci-bot", "Build #512 passed."),
    Message(5, "newsletter", "This week in ML: 10 papers you should read."),
    Message(6, "dave", "Thanks for the review!", read=True),
    Message(7, "alice", "Please review the deployment checklist before Friday."),
    Message(8, "ci-bot", "Build #513 passed."),
]


def recap_prompt(unread: list[Message], user: str) -> str:
    lines = "\n".join(f"[{m.id}] {m.sender}: {m.text}" for m in unread)
    return PromptTemplate.load("recap", "v1").render(user=user, messages=lines)


def summarize_unread(messages: list[Message], llm: LLM, user: str = "you", max_retries: int = 1) -> Recap:
    unread = [m for m in messages if not m.read]
    if not unread:
        return Recap("No unread messages.")
    prompt = recap_prompt(unread, user)
    for _ in range(max_retries + 1):
        reply = llm(prompt)
        try:
            data = parse_recap(reply)
        except InvalidModelReply as e:
            prompt = f"{prompt}\n\nYour previous reply was {e}. Reply again with only the JSON object."
            continue
        known = {m.id for m in unread}
        return Recap(data["summary"], [i for i in data.get("message_ids", []) if i in known])
    raise InvalidModelReply(f"no valid reply after {max_retries + 1} attempts")
