# 07 – Testing compound AI systems: prompts, RAG, agents

Slides: *"Modern AI Pipelines"*, *Deploying Prompt Templates*, *Retrieval
Augmented Generation (RAG)*, *AI Agents*, *Testing Compound AI System*.

A help center assistant and an email/calendar agent for a fictional food
delivery service. All LLM calls go through [litellm](https://docs.litellm.ai),
so any provider works.

| File | Content |
|---|---|
| `compound/llm.py` | `LiteLLM`: a callable `prompt -> text`; the model comes from `LLM_MODEL` |
| `compound/prompts.py`, `compound/prompts/*.txt` | Versioned prompt templates; screening of user input |
| `compound/rag.py`, `docs/` | TF-IDF retrieval over help center documents, then the LLM answers |
| `compound/tools.py` | Email and calendar tools (in memory); sending email needs user approval |
| `compound/agent.py` | ReAct agent: the LLM picks the next tool (inversion of control), with a step budget |
| `compound/evaluate.py` | Compares prompt versions on a small labeled question set with a real LLM |
| `tests/fakes.py` | `ScriptedLLM` (fixed replies) and `RuleBasedAgentLLM` (reacts to tool results) |

The tests replace the LLM with fakes, so they are fast, free, and
deterministic. They check the code *around* the model: prompts render, the
right context reaches the model, tool errors go back to the model, the loop
stops, unsafe actions need approval. Two retrieval tests are marked `xfail`:
lexical retrieval does not find paraphrased questions. Tests marked `llm` run
the slide's agent tests against a real model; they are skipped by default.

## Run

```sh
uv run pytest                       # fakes only, no API key needed
```

With a real model, set `LLM_MODEL` to any
[litellm model name](https://docs.litellm.ai/docs/providers) and the API key of
that provider (default: `anthropic/claude-opus-5`):

```sh
export LLM_MODEL=anthropic/claude-opus-5 ANTHROPIC_API_KEY=...
# or: export LLM_MODEL=openai/gpt-5-mini OPENAI_API_KEY=...
# or a local model: export LLM_MODEL=ollama/llama3.2   (needs a running Ollama)

uv run python -m compound.rag "How much is delivery for a \$15 order?"
uv run python -m compound.agent "What meetings do I have tomorrow?"
uv run python -m compound.evaluate                  # prompt v1 vs. v2
RUN_LLM_TESTS=1 uv run pytest -m llm                # tests against the real model
```
