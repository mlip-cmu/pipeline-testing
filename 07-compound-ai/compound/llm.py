"""LLM access through litellm, so that any provider can be used (set LLM_MODEL)."""

import os
from typing import Protocol

import litellm

DEFAULT_MODEL = "anthropic/claude-opus-5"


class LLM(Protocol):
    def __call__(self, prompt: str) -> str: ...


class LiteLLM:
    def __init__(self, model: str | None = None, system: str | None = None, **options):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.system = system
        self.options = options

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})
        response = litellm.completion(model=self.model, messages=messages, **self.options)
        return response.choices[0].message.content or ""
