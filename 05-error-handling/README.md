# 05 – Testing error handling and infrastructure robustness

Slides: *General Error Handling Strategies*, *Test Recovery Mechanisms with Stub*,
*Test Error Handling throughout Pipeline*, *Log Error Occurrence*,
*Example: Error Logging*, *Test Monitoring*.

(The *Test for Expected Exceptions* example with `NoDataError` is in
`../02-delivery-pipeline/tests/test_model.py`.)

| File | Content |
|---|---|
| `robustness/registry_client.py` | Download from the npm registry with `retry_call`; each retry increments a Prometheus counter |
| `robustness/pipeline_steps.py` | Load training data (unavailable, empty, stale, invalid data) and deploy a model (failed upload, corrupt upload) |
| `robustness/monitoring.py` | A prediction server, a monitor that watches its error count, a Slack notification service |
| `tests/test_retry.py` | `FailedConnection` stub injects timeouts: no problem, recoverable, not recoverable; retries are counted |
| `tests/test_pipeline_steps.py` | Stubs for the data source, model store, and notifier; errors are raised, repaired, logged, or reported |
| `tests/test_monitoring.py` | Stop the server, send requests, assert that the mock notification service got a message |
| `demo.py` | Real downloads with retries; metrics on http://localhost:8000/metrics |

## Run

```sh
uv run pytest
uv run python demo.py left-pad express     # then open http://localhost:8000/metrics
```
