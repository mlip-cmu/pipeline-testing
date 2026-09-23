import asyncio
import time

import pytest

from ensemble.predict import TooManyModelsFailed, predict_price

TIMEOUT_MS = 500
HOUSE = {"sqft": 1500, "bedrooms": 3}


def run(coroutine):
    start = time.perf_counter()
    try:
        return asyncio.run(coroutine)
    finally:
        run.elapsed_ms = (time.perf_counter() - start) * 1000


def test_average_of_all_models(model_apis):
    value = run(predict_price(HOUSE, [model_apis["M1"], model_apis["M2"]], TIMEOUT_MS))
    assert value == pytest.approx((50_000 + 180 * 1500 + 30_000 + 80_000 + 160 * 1500 + 45_000) / 2)


def test_success_despite_timeout(model_apis):
    value = run(predict_price(HOUSE, [model_apis["M1"], model_apis["M2"], model_apis["M3"]], TIMEOUT_MS))
    assert run.elapsed_ms < 2 * TIMEOUT_MS
    assert value > 0


def test_fail_on_too_many_timeouts(model_apis):
    with pytest.raises(TooManyModelsFailed):
        run(predict_price(HOUSE, [model_apis["M1"], model_apis["M3"], model_apis["M3"]], TIMEOUT_MS))
    assert run.elapsed_ms < 2 * TIMEOUT_MS


def test_unreachable_model_counts_as_failure(model_apis):
    unreachable = "http://127.0.0.1:9/predict"
    value = run(predict_price(HOUSE, [model_apis["M1"], model_apis["M2"], unreachable], TIMEOUT_MS))
    assert value > 0


def test_invalid_input_rejected_by_all_models(model_apis):
    with pytest.raises(TooManyModelsFailed):
        run(predict_price({"sqft": -5, "bedrooms": 3}, [model_apis["M1"], model_apis["M2"]], TIMEOUT_MS))
