# 06 – Test integration of components

Slides: *Test Integration of Components* (ensemble of models with timeouts),
*Integration and system tests*.

`predict_price` sends a house to several model services in parallel and returns
the average price. Slow or failing services are ignored. If fewer than two
services answer within the timeout, it raises `TooManyModelsFailed`.

| File | Content |
|---|---|
| `ensemble/model_server.py` | A FastAPI service for one price model (optional artificial delay) |
| `ensemble/predict.py` | The ensemble client (`asyncio` + `httpx`, like `Promise.all` on the slide) |
| `ensemble/background.py` | Starts a service on a free port in a background thread |
| `tests/conftest.py` | Launches three real model services (M3 is slow) before the tests, stops them after |
| `tests/test_ensemble.py` | "success despite timeout", "fail on too many timeouts", unreachable service, invalid input |

The tests use real HTTP calls to real servers, so they check the integration:
serialization, timeouts, and error handling together.

## Run

```sh
uv run pytest
uv run python demo.py
# or start a service yourself and open http://localhost:3001/docs
MODEL_NAME=neighborhood MODEL_DELAY_S=1 uv run uvicorn ensemble.model_server:app --port 3001
```
