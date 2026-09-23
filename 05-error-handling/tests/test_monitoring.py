import threading

import pytest

from robustness.monitoring import Monitor, PredictionServer, ServiceUnavailable


class MyNotificationService:
    def __init__(self):
        self.messages = []
        self.received_notification = threading.Event()

    def send_notification(self, message):
        self.messages.append(message)
        self.received_notification.set()


def send_requests(server, n):
    for _ in range(n):
        try:
            server.request({"x": 1})
        except ServiceUnavailable:
            pass


@pytest.fixture
def server():
    return PredictionServer(model=lambda features: 42)


def test_monitor_notifies_when_server_is_down(server):
    notifications = MyNotificationService()
    monitor = Monitor(server, notifications, interval=0.05).start()
    server.stop()
    send_requests(server, 2)
    assert notifications.received_notification.wait(timeout=1)
    monitor.stop()


def test_monitor_stays_quiet_when_server_is_healthy(server):
    notifications = MyNotificationService()
    monitor = Monitor(server, notifications, interval=0.05).start()
    send_requests(server, 20)
    assert not notifications.received_notification.wait(timeout=0.3)
    monitor.stop()


def test_single_failure_is_tolerated(server):
    notifications = MyNotificationService()
    monitor = Monitor(server, notifications, interval=10)
    server.stop()
    send_requests(server, 1)
    monitor.check()
    assert notifications.messages == []
    send_requests(server, 5)
    monitor.check()
    assert "5 failed requests" in notifications.messages[0]
