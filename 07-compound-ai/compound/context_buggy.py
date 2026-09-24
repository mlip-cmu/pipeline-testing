"""The context assembly code from the slide "Subtle Bugs in Context Assembly Code", same interface as context.py."""

import json

from jinja2 import Template

from compound.context import MAX_MESSAGES, count_tokens, message  # noqa: F401


def build_context(system, history, token_budget):
    context = history[:MAX_MESSAGES]
    return [system] + context


def build_context_tail(system, history, token_budget):
    return ([system] + history)[-MAX_MESSAGES:]


def render(template, **values):
    return Template(template).render(**values)


def fits(prompt, token_budget):
    return not len(prompt) > 8000


def parse_recap(reply):
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        return {"summary": ""}


def call_tool(tool, args):
    try:
        return json.dumps(tool(**args), default=str)
    except Exception:
        return ""
