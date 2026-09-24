# Automating and Testing ML Pipelines – code examples

Working Python code for the lecture *Automating and Testing ML Pipelines*
(Machine Learning in Production). Each directory is a separate, small project
that expands on the code snippets from the slides.

| Project | Slides | Topic |
|---|---|---|
| [`01-wrangling-bugs`](01-wrangling-bugs) | Subtle Bugs in Data Wrangling Code | A notebook with silent bugs in pandas code; tests that miss or find them |
| [`02-delivery-pipeline`](02-delivery-pipeline) | Pipelines are Code, Test the Modules, Orchestrating Functions, End-To-End Test, Tracking Model Qualities | Notebook → modular pipeline; unit, integration, and end-to-end tests; DVC; MLflow |
| [`03-ml-test-score`](03-ml-test-score) | ML Test Score, Case Study: Covid-19 Detection | Data, model, infrastructure, and monitoring tests from the rubric |
| [`04-stubbing-dependencies`](04-stubbing-dependencies) | Testing across boundaries | Decouple code from an external API; stubs and mocks |
| [`05-error-handling`](05-error-handling) | Test Recovery Mechanisms with Stub, Error Logging, Test Monitoring | Retries, fault injection with stubs, error logging with Prometheus, monitor tests |
| [`06-component-integration`](06-component-integration) | Test Integration of Components | Ensemble of model services with timeouts, tested against real HTTP servers |
| [`07-compound-ai`](07-compound-ai) | Unit Tests with LLM Calls, Record and Replay, Subtle Bugs in Context Assembly Code, Agents, Inject Model API Faults | Stubbed and recorded LLMs, context assembly bugs, fallback and idempotent retries, agent harness tests; any LLM via litellm |
| [`08-meeting-agent`](08-meeting-agent) | Breakout: Testing a Meeting Agent | A LangGraph agent that updates GitHub issues from meeting transcripts; no tests, as an exercise |

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS and Linux
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uv installs Python 3.12 and all dependencies of a project the first time you
run a command in its directory. Each project has its own `pyproject.toml` and a
`uv.lock` with the exact versions.

```sh
cd 02-delivery-pipeline
uv run pytest                     # run the tests
uv run python -m delivery.pipeline
```

The README in each project lists its commands. Some projects have optional
dependency groups (for example Jupyter, DVC, and MLflow in `02`), which you
enable with `uv run --group <name> ...`.

To run all tests (`08-meeting-agent` has no tests, on purpose):

```sh
for p in 0[1-7]*/; do (cd "$p" && uv run pytest -q); done
```

## Continuous integration

`.github/workflows/ci.yml` runs the tests of all projects on every push. A
second job runs the DVC pipeline and the notebook of `02-delivery-pipeline`.

A third job checks that the graphs of `08-meeting-agent` build.

Only `07-compound-ai` and `08-meeting-agent` use an LLM. The tests of
`07-compound-ai` use fake LLMs; tests against a real model are optional (see
its README).
