import threading

import pytest

from robustness.monitoring import Monitor, PredictionServer, ServiceUnavailable


class FakeNotifier:
    def __init__(self):
        self.messages = []
        self.received_notification = threading.Event()

    def send(self, msg):
        self.messages.append(msg)
        self.received_notification.set()


class FakeServer(PredictionServer):
    def __init__(self):
        super().__init__(model=lambda features: 42)


def send_requests(server, n):
    for _ in range(n):
        try:
            server.request({"x": 1})
        except ServiceUnavailable:
            pass


@pytest.fixture
def server():
    return FakeServer()


def test_alert_when_server_down():
    server = FakeServer()
    notifier = FakeNotifier()
    monitor = Monitor(server, notifier)
    server.stop()
    send_requests(server, 2)
    monitor.check()
    assert len(notifier.messages) == 1


def test_monitor_notifies_when_server_is_down(server):
    notifications = FakeNotifier()
    monitor = Monitor(server, notifications, interval=0.05).start()
    server.stop()
    send_requests(server, 2)
    assert notifications.received_notification.wait(timeout=1)
    monitor.stop()


def test_monitor_stays_quiet_when_server_is_healthy(server):
    notifications = FakeNotifier()
    monitor = Monitor(server, notifications, interval=0.05).start()
    send_requests(server, 20)
    assert not notifications.received_notification.wait(timeout=0.3)
    monitor.stop()


def test_single_failure_is_tolerated(server):
    notifications = FakeNotifier()
    monitor = Monitor(server, notifications, interval=10)
    server.stop()
    send_requests(server, 1)
    monitor.check()
    assert notifications.messages == []
    send_requests(server, 5)
    monitor.check()
    assert "5 failed requests" in notifications.messages[0]
