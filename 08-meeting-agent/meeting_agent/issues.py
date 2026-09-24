"""Access to the team's GitHub issues."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import httpx


@dataclass
class Issue:
    number: int
    title: str
    state: str
    assignees: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    body: str = ""
    updated_at: datetime | None = None


class IssueTracker(Protocol):
    def list_issues(self, state: str = "open") -> list[Issue]: ...
    def get_issue(self, number: int) -> Issue | None: ...
    def close_issue(self, number: int, comment: str | None = None) -> None: ...
    def assign(self, number: int, login: str) -> None: ...
    def create_issue(self, title: str, body: str, assignees: list[str]) -> Issue: ...
    def comment(self, number: int, text: str) -> None: ...


class GitHubIssueTracker:
    def __init__(self, repo: str, token: str | None = None, base_url: str = "https://api.github.com"):
        self.repo = repo
        self.http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token or os.environ['GITHUB_TOKEN']}",
                     "Accept": "application/vnd.github+json"},
            timeout=10,
        )

    def _issue(self, data: dict) -> Issue:
        return Issue(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            assignees=[a["login"] for a in data.get("assignees", [])],
            labels=[label["name"] for label in data.get("labels", [])],
            body=data.get("body") or "",
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )

    def list_issues(self, state: str = "open") -> list[Issue]:
        response = self.http.get(f"/repos/{self.repo}/issues", params={"state": state, "per_page": 100})
        response.raise_for_status()
        return [self._issue(i) for i in response.json() if "pull_request" not in i]

    def get_issue(self, number: int) -> Issue | None:
        response = self.http.get(f"/repos/{self.repo}/issues/{number}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return self._issue(response.json())

    def close_issue(self, number: int, comment: str | None = None) -> None:
        if comment:
            self.comment(number, comment)
        self.http.patch(f"/repos/{self.repo}/issues/{number}", json={"state": "closed"}).raise_for_status()

    def assign(self, number: int, login: str) -> None:
        self.http.post(f"/repos/{self.repo}/issues/{number}/assignees", json={"assignees": [login]}).raise_for_status()

    def create_issue(self, title: str, body: str, assignees: list[str]) -> Issue:
        response = self.http.post(f"/repos/{self.repo}/issues",
                                  json={"title": title, "body": body, "assignees": assignees})
        response.raise_for_status()
        return self._issue(response.json())

    def comment(self, number: int, text: str) -> None:
        self.http.post(f"/repos/{self.repo}/issues/{number}/comments", json={"body": text}).raise_for_status()
