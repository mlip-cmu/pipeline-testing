"""Sending email with retries, without sending it twice."""

import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    body: str


class Outbox(Protocol):
    def send(self, email: Email, idempotency_key: str) -> str: ...


def send_with_retry(outbox: Outbox, email: Email, idempotency_key: str | None = None, attempts: int = 3) -> str:
    key = idempotency_key or str(uuid.uuid4())
    for attempt in range(attempts):
        try:
            return outbox.send(email, idempotency_key=key)
        except TimeoutError:
            if attempt + 1 == attempts:
                raise
    raise AssertionError("unreachable")
