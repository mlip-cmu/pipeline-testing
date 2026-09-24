"""A ReAct-style agent: the LLM decides which tool to call next (inversion of control)."""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from compound.llm import LLM, LiteLLM
from compound.prompts import PromptTemplate, screen_user_input
from compound.tools import TOOL_DESCRIPTIONS, sample_workspace

ACTION = re.compile(r"^\s*ACTION:\s*(\w+)\s*(\{.*\})?\s*$", re.S)
FINAL = re.compile(r"FINAL ANSWER:\s*(.*)", re.S)


@dataclass
class AgentResult:
    answer: str
    terminated: bool
    steps_taken: int
    transcript: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)


def is_final_answer(reply: str) -> bool:
    return FINAL.search(reply) is not None


def run_tool(tools: dict[str, Callable], reply: str) -> tuple[str, str | None]:
    """Returns the observation for the model and, if the tool call failed, an error for the harness."""
    match = ACTION.match(reply)
    if not match:
        return "Error: could not parse the reply. Use 'ACTION: <tool> <json>' or 'FINAL ANSWER: <answer>'.", None
    name, arguments = match.group(1), match.group(2) or "{}"
    if name not in tools:
        return f"Error: unknown tool {name}. Available tools: {', '.join(tools)}", f"{name}: unknown tool"
    try:
        return json.dumps(tools[name](**json.loads(arguments)), default=str), None
    except Exception as e:
        return f"Error: {name} failed: {type(e).__name__}: {e}", f"{name}: {type(e).__name__}: {e}"


def react_agent(query: str, tools: dict[str, Callable] | None = None, max_steps: int = 10,
                llm: LLM | None = None, today: date | None = None) -> AgentResult:
    llm = llm or LiteLLM()
    today = today or date.today()
    tools = {**sample_workspace(today).tools(), **(tools or {})}
    template = PromptTemplate.load("agent", "v1")
    descriptions = "\n".join(TOOL_DESCRIPTIONS.get(name, name) for name in tools)
    transcript = [f"Question: {screen_user_input(query)}"]
    errors: list[str] = []

    for step in range(1, max_steps + 1):
        prompt = template.render(today=today.isoformat(), tools=descriptions, transcript="\n".join(transcript))
        reply = llm(prompt).strip()
        if is_final_answer(reply):
            transcript.append(reply)
            return AgentResult(FINAL.search(reply).group(1).strip(), True, step, transcript, errors)
        observation, error = run_tool(tools, reply)
        if error:
            errors.append(error)
        transcript += [reply, f"Observation: {observation}"]

    return AgentResult("Sorry, I could not complete this request.", False, max_steps, transcript, errors)


if __name__ == "__main__":
    import sys

    result = react_agent(" ".join(sys.argv[1:]) or "What meetings do I have tomorrow?")
    print("\n".join(result.transcript))
    print(f"\nAnswer ({result.steps_taken} steps): {result.answer}")
