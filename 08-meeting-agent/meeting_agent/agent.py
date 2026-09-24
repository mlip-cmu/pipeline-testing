"""LangGraph workflows: process a meeting transcript, and generate a meeting agenda."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from meeting_agent.issues import Issue, IssueTracker
from meeting_agent.prompts import AGENDA, PROCESS_TRANSCRIPT
from meeting_agent.tools import ActionLog, make_tools
from meeting_agent.transcripts import format_transcript, load_transcript

MAX_ISSUES_IN_CONTEXT = 50


def default_llm():
    return ChatLiteLLM(model=os.environ.get("LLM_MODEL", "anthropic/claude-opus-5"), max_retries=2)


def issue_lines(issues: list[Issue]) -> str:
    return "\n".join(f"#{i.number} {i.title} (assigned: {', '.join(i.assignees) or 'nobody'})"
                     for i in issues[:MAX_ISSUES_IN_CONTEXT]) or "(none)"


# ---------------------------------------------------------------- transcript -> issue updates

class TranscriptState(MessagesState):
    transcript: str
    report: str


def build_transcript_graph(llm, tracker: IssueTracker, team: dict[str, str], tools):
    model = llm.bind_tools(tools)

    def load_context(state: TranscriptState):
        system = PROCESS_TRANSCRIPT.format(team=", ".join(team), issues=issue_lines(tracker.list_issues("open")))
        return {"messages": [SystemMessage(system), HumanMessage(f"Transcript:\n{state['transcript']}")]}

    def agent(state: TranscriptState):
        return {"messages": [model.invoke(state["messages"])]}

    def write_report(state: TranscriptState):
        last = state["messages"][-1]
        return {"report": last.content if isinstance(last, AIMessage) else ""}

    graph = StateGraph(TranscriptState)
    graph.add_node("load_context", load_context)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("write_report", write_report)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "write_report"})
    graph.add_edge("tools", "agent")
    graph.add_edge("write_report", END)
    return graph.compile()


@dataclass
class MeetingResult:
    report: str
    actions: ActionLog


def process_transcript(transcript_path: str | Path, tracker: IssueTracker, team: dict[str, str],
                       llm=None, dry_run: bool = True, max_steps: int = 25) -> MeetingResult:
    transcript = format_transcript(load_transcript(transcript_path))
    tools, log = make_tools(tracker, team, dry_run=dry_run)
    graph = build_transcript_graph(llm or default_llm(), tracker, team, tools)
    state = graph.invoke({"messages": [], "transcript": transcript, "report": ""},
                         config={"recursion_limit": max_steps})
    return MeetingResult(state["report"], log)


# ---------------------------------------------------------------- issues -> agenda

class AgendaState(TypedDict, total=False):
    today: datetime
    stale_days: int
    issues: list[Issue]
    stale: list[Issue]
    draft: str
    agenda: str


def build_agenda_graph(llm, tracker: IssueTracker):
    def fetch_issues(state: AgendaState):
        return {"issues": tracker.list_issues("open")}

    def find_stale(state: AgendaState):
        cutoff = state["today"] - timedelta(days=state["stale_days"])
        return {"stale": [i for i in state["issues"] if i.updated_at and i.updated_at < cutoff]}

    def draft(state: AgendaState):
        prompt = AGENDA.format(today=state["today"].date(), stale_days=state["stale_days"],
                               issues=issue_lines(state["issues"]), stale=issue_lines(state["stale"]))
        return {"draft": llm.invoke([HumanMessage(prompt)]).content}

    def add_links(state: AgendaState):
        known = {i.number for i in state["issues"]}
        links = "\n".join(f"- #{n}: https://github.com/{getattr(tracker, 'repo', '')}/issues/{n}" for n in sorted(known))
        return {"agenda": f"{state['draft']}\n\nIssues:\n{links}"}

    graph = StateGraph(AgendaState)
    for name, node in [("fetch_issues", fetch_issues), ("find_stale", find_stale), ("draft", draft), ("add_links", add_links)]:
        graph.add_node(name, node)
    graph.add_edge(START, "fetch_issues")
    graph.add_edge("fetch_issues", "find_stale")
    graph.add_edge("find_stale", "draft")
    graph.add_edge("draft", "add_links")
    graph.add_edge("add_links", END)
    return graph.compile()


def generate_agenda(tracker: IssueTracker, llm=None, today: datetime | None = None, stale_days: int = 14) -> str:
    graph = build_agenda_graph(llm or default_llm(), tracker)
    state = graph.invoke({"today": today or datetime.now(timezone.utc), "stale_days": stale_days})
    return state["agenda"]


def load_team(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())
