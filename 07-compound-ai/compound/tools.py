"""Email and calendar tools the agent can call (in-memory stand-ins for real APIs)."""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable


@dataclass
class Email:
    sender: str
    to: str
    subject: str
    body: str
    day: date


@dataclass
class Event:
    title: str
    day: date
    start: str
    end: str


@dataclass
class Workspace:
    emails: list[Email] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    outbox: list[Email] = field(default_factory=list)
    user: str = "me@example.com"
    confirm_send: Callable[[Email], bool] = lambda email: True

    def get_email(self, start: str, end: str) -> list[dict]:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        return [{"from": e.sender, "subject": e.subject, "day": e.day.isoformat(), "body": e.body}
                for e in self.emails if first <= e.day <= last]

    def get_calendar(self, start: str, end: str) -> list[dict]:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        return [{"title": e.title, "day": e.day.isoformat(), "start": e.start, "end": e.end}
                for e in self.events if first <= e.day <= last]

    def send_email(self, to: str, subject: str, body: str) -> str:
        email = Email(self.user, to, subject, body, date.today())
        if not self.confirm_send(email):
            return "The user did not approve sending this email."
        self.outbox.append(email)
        return f"Email sent to {to}."

    def tools(self) -> dict:
        return {"get_email": self.get_email, "get_calendar": self.get_calendar, "send_email": self.send_email}


TOOL_DESCRIPTIONS = {
    "get_email": 'get_email {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} - emails received in a date range',
    "get_calendar": 'get_calendar {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} - meetings in a date range',
    "send_email": 'send_email {"to": "...", "subject": "...", "body": "..."} - send an email (the user must approve)',
}


def sample_workspace(today: date) -> Workspace:
    tomorrow = date.fromordinal(today.toordinal() + 1)
    return Workspace(
        emails=[
            Email("alice@example.com", "me@example.com", "Project sync", "Can we move our sync to 3pm tomorrow?", today),
            Email("bob@example.com", "me@example.com", "Lunch", "Lunch on Friday?", today),
        ],
        events=[
            Event("Team standup", tomorrow, "09:30", "09:45"),
            Event("Project sync with Alice", tomorrow, "14:00", "15:00"),
        ],
    )
