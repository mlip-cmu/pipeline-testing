"""Versioned prompt templates stored as files, and screening of user input."""

import re
import string
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
MAX_INPUT_CHARS = 2_000
SUSPICIOUS = re.compile(r"ignore (all |any )?(previous|prior|above) instructions|system prompt|</?context>", re.I)


class RejectedInput(ValueError):
    pass


def screen_user_input(text: str) -> str:
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t").strip()
    if not text:
        raise RejectedInput("Empty input")
    if len(text) > MAX_INPUT_CHARS:
        raise RejectedInput(f"Input longer than {MAX_INPUT_CHARS} characters")
    if SUSPICIOUS.search(text):
        raise RejectedInput("Input looks like a prompt injection")
    return text


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: string.Template

    @classmethod
    def load(cls, name: str, version: str = "v1") -> "PromptTemplate":
        text = (PROMPT_DIR / f"{name}_{version}.txt").read_text()
        return cls(name, version, string.Template(text))

    @property
    def variables(self) -> set[str]:
        return set(self.template.get_identifiers())

    def render(self, **values: str) -> str:
        missing = self.variables - values.keys()
        if missing:
            raise KeyError(f"Missing template variables: {sorted(missing)}")
        return self.template.substitute(values)
