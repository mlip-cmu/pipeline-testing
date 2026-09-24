"""Tools that the agent can call to read and change issues."""

import json
from dataclasses import dataclass, field

from langchain_core.tools import tool

from meeting_agent.issues import IssueTracker


@dataclass
class ActionLog:
    applied: list[dict] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)


def make_tools(tracker: IssueTracker, team: dict[str, str], dry_run: bool = True):
    log = ActionLog()

    def record(action: dict) -> str:
        log.applied.append({**action, "dry_run": dry_run})
        return ("Planned (dry run): " if dry_run else "Done: ") + json.dumps(action)

    def refuse(action: dict, reason: str) -> str:
        log.refused.append({**action, "reason": reason})
        return f"Error: {reason}. Do not retry; mention this in your report."

    def resolve(person: str) -> str | None:
        return team.get(person) or team.get(person.strip().title()) or (person if person in team.values() else None)

    @tool
    def list_open_issues() -> str:
        """List all open issues with number, title, and assignees."""
        return json.dumps([{"number": i.number, "title": i.title, "assignees": i.assignees}
                           for i in tracker.list_issues("open")])

    @tool
    def get_issue(number: int) -> str:
        """Get the details of one issue."""
        issue = tracker.get_issue(number)
        if issue is None:
            return f"Error: issue #{number} does not exist"
        return json.dumps({"number": issue.number, "title": issue.title, "state": issue.state,
                           "assignees": issue.assignees, "body": issue.body[:2000]})

    @tool
    def close_issue(number: int, reason: str) -> str:
        """Close an issue that the team agreed is done. Give the reason from the meeting."""
        action = {"action": "close", "issue": number, "reason": reason}
        issue = tracker.get_issue(number)
        if issue is None:
            return refuse(action, f"issue #{number} does not exist")
        if issue.state == "closed":
            return refuse(action, f"issue #{number} is already closed")
        if not dry_run:
            tracker.close_issue(number, comment=f"Closed after team meeting: {reason}")
        return record(action)

    @tool
    def assign_issue(number: int, person: str) -> str:
        """Assign an issue to a team member, using the person's name from the meeting."""
        action = {"action": "assign", "issue": number, "person": person}
        login = resolve(person)
        if login is None:
            return refuse(action, f"{person} is not a team member ({', '.join(team)})")
        if tracker.get_issue(number) is None:
            return refuse(action, f"issue #{number} does not exist")
        if not dry_run:
            tracker.assign(number, login)
        return record({**action, "login": login})

    @tool
    def create_issue(title: str, description: str, assignee: str | None = None) -> str:
        """Create a new issue for a task that the team agreed on and that has no issue yet."""
        action = {"action": "create", "title": title, "assignee": assignee}
        login = resolve(assignee) if assignee else None
        if assignee and login is None:
            return refuse(action, f"{assignee} is not a team member ({', '.join(team)})")
        if not dry_run:
            issue = tracker.create_issue(title, description + "\n\n_Created from meeting notes._", [login] if login else [])
            action["issue"] = issue.number
        return record(action)

    @tool
    def comment_on_issue(number: int, comment: str) -> str:
        """Add a comment with progress or decisions from the meeting to an existing issue."""
        action = {"action": "comment", "issue": number, "comment": comment}
        if tracker.get_issue(number) is None:
            return refuse(action, f"issue #{number} does not exist")
        if not dry_run:
            tracker.comment(number, comment)
        return record(action)

    tools = [list_open_issues, get_issue, close_issue, assign_issue, create_issue, comment_on_issue]
    return tools, log
