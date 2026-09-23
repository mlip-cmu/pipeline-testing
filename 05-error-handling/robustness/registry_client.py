"""Download package metadata from the npm registry, with retries on timeouts."""

import httpx
from prometheus_client import Counter
from retry.api import retry_call

REGISTRY_URL = "https://registry.npmjs.org/"

connection_retry_counter = Counter("connection_retry_total", "Retry attempts on failed connections")


class Connection:
    def __init__(self, timeout: float = 2.0):
        self.http = httpx.Client(timeout=timeout, follow_redirects=True)

    def get(self, url: str) -> str:
        try:
            response = self.http.get(url)
        except httpx.TimeoutException as e:
            raise TimeoutError(str(e)) from e
        response.raise_for_status()
        return response.text


class RetryLogger:
    def warning(self, fmt, error, delay):
        connection_retry_counter.inc()


retry_logger = RetryLogger()


def get_data(connection: Connection, value: str) -> str:
    def get():
        return connection.get(REGISTRY_URL + value)
    return retry_call(get, exceptions=TimeoutError, tries=3, delay=0.1, backoff=2, logger=retry_logger)
