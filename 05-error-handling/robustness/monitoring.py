"""A prediction server and a monitor that notifies the team when requests fail."""

import threading
from typing import Protocol

import httpx


class ServiceUnavailable(Exception):
    pass


class NotificationService(Protocol):
    def send_notification(self, message: str) -> None: ...


class SlackNotificationService:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_notification(self, message: str) -> None:
        httpx.post(self.webhook_url, json={"text": message}, timeout=5).raise_for_status()


class PredictionServer:
    def __init__(self, model):
        self.model = model
        self.running = True
        self.requests_total = 0
        self.errors_total = 0
        self._lock = threading.Lock()

    def stop(self) -> None:
        self.running = False

    def request(self, features):
        with self._lock:
            self.requests_total += 1
            if not self.running:
                self.errors_total += 1
        if not self.running:
            raise ServiceUnavailable("server stopped")
        return self.model(features)

    def metrics(self) -> dict:
        with self._lock:
            return {"requests_total": self.requests_total, "errors_total": self.errors_total}


class Monitor:
    def __init__(self, server: PredictionServer, notifications: NotificationService,
                 interval: float = 1.0, max_errors_per_interval: int = 1):
        self.server = server
        self.notifications = notifications
        self.interval = interval
        self.max_errors = max_errors_per_interval
        self._stop = threading.Event()
        self._last_errors = server.metrics()["errors_total"]
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Monitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def check(self) -> None:
        errors = self.server.metrics()["errors_total"]
        new_errors, self._last_errors = errors - self._last_errors, errors
        if new_errors > self.max_errors:
            self.notifications.send_notification(f"{new_errors} failed requests in the last {self.interval}s")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.check()
