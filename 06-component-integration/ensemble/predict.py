"""Predict a price as the average of several model services; tolerate slow or failing services."""

import asyncio

import httpx


class TooManyModelsFailed(Exception):
    pass


def parse_result(response: httpx.Response) -> float:
    response.raise_for_status()
    return float(response.json()["price"])


async def _rpc(client: httpx.AsyncClient, url: str, data: dict, timeout_s: float) -> float:
    try:
        return parse_result(await client.post(url, json=data, timeout=timeout_s))
    except (httpx.HTTPError, KeyError, ValueError):
        return -1


async def predict_price(data: dict, models: list[str], timeout_ms: float) -> float:
    async with httpx.AsyncClient() as client:
        predictions = await asyncio.gather(*(_rpc(client, url, data, timeout_ms / 1000) for url in models))
    success = [p for p in predictions if p >= 0]
    if len(success) < 2:
        raise TooManyModelsFailed(f"Only {len(success)} of {len(models)} models answered")
    return sum(success) / len(success)
