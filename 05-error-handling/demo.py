"""Fetch npm package metadata with retries and expose retry counts as Prometheus metrics."""

import sys
import time

import httpx
from prometheus_client import start_http_server

from robustness.registry_client import Connection, get_data

if __name__ == "__main__":
    start_http_server(8000)
    print("Metrics at http://localhost:8000/metrics (Ctrl+C to stop)")
    for package in sys.argv[1:] or ["left-pad", "express"]:
        try:
            print(package, get_data(Connection(timeout=0.5), package)[:80], "...")
        except TimeoutError as e:
            print(package, "failed after 3 attempts:", e)
        except httpx.HTTPError as e:
            print(package, "failed without retry:", e)
    while True:
        time.sleep(1)
