"""Context assembly and result parsing for LLM calls."""

import json
import os
from pathlib import Path

import jinja2
import litellm

MAX_MESSAGES = 20
PROMPTS = Path(__file__).parent / "prompts"


class InvalidModelReply(ValueError):
    pass


def message(text: str, role: str = "user") -> dict:
    return {"role": role, "content": text}


def count_tokens(messages: list[dict]) -> int:
    return litellm.token_counter(model=os.environ.get("LLM_MODEL", "anthropic/claude-opus-5"), messages=messages)


def build_context(system: dict, history: list[dict], token_budget: int) -> list[dict]:
    kept: list[dict] = []
    for msg in reversed(history):
        if count_tokens([system, msg, *kept]) > token_budget:
            break
        kept.insert(0, msg)
    return [system, *kept]


def render(template: str, **values) -> str:
    return jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(template).render(**values)


def load_prompt(name: str, **values) -> str:
    return render((PROMPTS / f"{name}.j2").read_text(), **values)


def fits(prompt: str, token_budget: int) -> bool:
    return count_tokens([message(prompt)]) <= token_budget


def parse_recap(reply: str) -> dict:
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as e:
        raise InvalidModelReply(f"invalid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
        raise InvalidModelReply("invalid JSON: expected an object with a string 'summary'")
    return data


def call_tool(tool, args: dict) -> str:
    try:
        return json.dumps(tool(**args), default=str)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
