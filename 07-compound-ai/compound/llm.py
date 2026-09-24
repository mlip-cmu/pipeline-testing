"""LLM access through litellm, so that any provider can be used (set LLM_MODEL), plus retries and fallback."""

import logging
import os
import time
from typing import Callable, Protocol

import litellm

DEFAULT_MODEL = "anthropic/claude-opus-5"
log = logging.getLogger(__name__)


class LLM(Protocol):
    def __call__(self, prompt: str) -> str: ...


class RateLimitError(Exception):
    def __init__(self, retry_after: float = 1.0):
        super().__init__(f"rate limited, retry after {retry_after}s")
        self.retry_after = retry_after


class ModelNotFound(Exception):
    pass


class AllModelsFailed(Exception):
    def __init__(self, errors: list[Exception]):
        super().__init__(f"all models failed: {errors}")
        self.errors = errors


def _retry_after(error: Exception) -> float:
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after", 1))
    except ValueError:
        return 1.0


class LiteLLM:
    def __init__(self, model: str | None = None, system: str | None = None, **options):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.system = system
        self.options = options

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})
        try:
            response = litellm.completion(model=self.model, messages=messages, **self.options)
        except litellm.RateLimitError as e:
            raise RateLimitError(_retry_after(e)) from e
        except litellm.Timeout as e:
            raise TimeoutError(str(e)) from e
        except litellm.NotFoundError as e:
            raise ModelNotFound(self.model) from e
        return response.choices[0].message.content or ""


class LLMClient:
    """Tries each model in order; retries rate limits and timeouts; counts fallbacks so they are not silent."""

    def __init__(self, models: list[LLM], max_attempts: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.models = models
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.fallbacks = 0

    def complete(self, prompt: str) -> str:
        errors: list[Exception] = []
        for index, model in enumerate(self.models):
            if index > 0:
                self.fallbacks += 1
                log.warning("Falling back to model %d after: %s", index, errors[-1])
            for attempt in range(self.max_attempts):
                try:
                    return model(prompt)
                except RateLimitError as e:
                    errors.append(e)
                    if attempt + 1 < self.max_attempts:
                        self.sleep(e.retry_after)
                except TimeoutError as e:
                    errors.append(e)
                    if attempt + 1 < self.max_attempts:
                        self.sleep(2 ** attempt)
                except ModelNotFound as e:
                    errors.append(e)
                    break
        raise AllModelsFailed(errors)

    __call__ = complete
