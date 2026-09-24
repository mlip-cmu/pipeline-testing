# 07 – Testing compound AI systems

Slides: *Unit Tests with LLM Calls*, *Stubbing the Model*, *Record and Replay
Model Calls*, *Test Strategies for Code that Calls LLMs*, *Subtle Bugs in
Context Assembly Code*, *Agents: Test the Harness, not the Model*,
*Fail-Plausible*, *Inject Model API Faults with a Stub*, *Testing Error
Handling in Compound AI System*, *Test Context Assembly and Result Parsing*,
*Integration and system tests*.

A chat service with a recap of unread messages, and an agent with email and
calendar tools. All LLM calls go through [litellm](https://docs.litellm.ai), so
any provider works. The tests use no real model, except the smoke test marked
`llm`.

| File | Content |
|---|---|
| `compound/recap.py` | Recap of unread messages: the model answers in JSON, invalid JSON is sent back once |
| `compound/context.py` | Context assembly: token budget, strict prompt templates, result parsing, tool errors |
| `compound/context_buggy.py` | The buggy snippets from the slide, with the same interface |
| `compound/app.py` | FastAPI chat service with `/send`, `/read`, `/recap` |
| `compound/replay.py`, `compound/record.py`, `recordings/` | Record model replies once, replay them in tests |
| `compound/llm.py` | `LiteLLM` (model from `LLM_MODEL`); `LLMClient` with retries, `retry-after`, and a counted fallback |
| `compound/outbox.py` | Send email with retries and one idempotency key |
| `compound/agent.py`, `compound/tools.py` | ReAct agent; tool errors are recorded in the result, sending email needs approval |
| `compound/prompts/` | Prompt templates (Jinja, rendered with `StrictUndefined`) |
| `tests/fakes.py` | `ScriptedLLM`, `FlakyModel`, `FakeOutbox`, and a rule-based fake model for the agent |

Tests by layer (slide *Test Strategies for Code that Calls LLMs*):

| Layer | Model replaced by | Tests |
|---|---|---|
| Unit and harness tests | `ScriptedLLM` and other fakes | `test_recap.py`, `test_context.py`, `test_agent.py`, `test_llm_faults.py` |
| Integration and system tests | Recorded replies (`ReplayLLM`) | `test_recap.py::test_recap_with_recorded_reply`, `test_app.py` |
| Live smoke tests | Real model | `test_llm.py` (marked `llm`, skipped by default) |

In `test_context.py`, every buggy snippet from the slide is marked `xfail`:
the tests catch all of them.

**Note:** `recordings/recap.json` has *hand-written* example replies, so that
the tests run without an API key. Record real replies with
`uv run python -m compound.record` (this needs an API key; see below).

## Run

```sh
uv run pytest                       # no API key needed
```

With a real model, set `LLM_MODEL` to any
[litellm model name](https://docs.litellm.ai/docs/providers) and the API key of
that provider (default: `anthropic/claude-opus-5`):

```sh
export LLM_MODEL=anthropic/claude-opus-5 ANTHROPIC_API_KEY=...
# or: export LLM_MODEL=openai/gpt-5-mini OPENAI_API_KEY=...
# or a local model: export LLM_MODEL=ollama/llama3.2   (needs a running Ollama)

uv run python -m compound.record                    # record real replies for the replay tests
uv run uvicorn compound.app:app                     # chat service, open http://localhost:8000/docs
uv run python -m compound.agent "What meetings do I have tomorrow?"
RUN_LLM_TESTS=1 uv run pytest -m llm                # live smoke test
```

A real model can give a different recap than the recording. If you record
again, the replay tests can fail at `message_ids == [3, 7]`. This shows that
recordings fix one model answer, and that an assertion on a model answer is a
decision about what is acceptable.
