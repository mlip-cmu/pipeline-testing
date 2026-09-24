"""Command line: python -m meeting_agent {agenda,process,graph} ..."""

import argparse
import json

from meeting_agent.agent import build_agenda_graph, build_transcript_graph, default_llm, generate_agenda, load_team, process_transcript
from meeting_agent.issues import GitHubIssueTracker
from meeting_agent.tools import make_tools


def main():
    parser = argparse.ArgumentParser(prog="meeting_agent")
    parser.add_argument("--repo", default="mlip-cmu/team-project", help="GitHub repository owner/name")
    parser.add_argument("--team", default="data/team.json", help="JSON file mapping names to GitHub logins")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("agenda", help="generate the agenda for the next meeting")
    process = sub.add_parser("process", help="update issues from a meeting transcript")
    process.add_argument("transcript")
    process.add_argument("--apply", action="store_true", help="change the issues (default: dry run)")
    sub.add_parser("graph", help="print both graphs as Mermaid diagrams (no GitHub or LLM access)")
    args = parser.parse_args()

    team = load_team(args.team)
    if args.command == "graph":
        tools, _ = make_tools(tracker=None, team=team)
        print(build_transcript_graph(default_llm(), None, team, tools).get_graph().draw_mermaid())
        print(build_agenda_graph(default_llm(), None).get_graph().draw_mermaid())
        return

    tracker = GitHubIssueTracker(args.repo)
    if args.command == "agenda":
        print(generate_agenda(tracker))
    else:
        result = process_transcript(args.transcript, tracker, team, dry_run=not args.apply)
        print(result.report)
        print(json.dumps({"applied": result.actions.applied, "refused": result.actions.refused}, indent=2))


if __name__ == "__main__":
    main()
