import pytest
from prometheus_client import REGISTRY

from robustness.registry_client import Connection, get_data


class FailedConnection(Connection):
    """Stub of a network connection that times out a given number of times."""

    def __init__(self, failures):
        self.remaining_failures = failures
        self.calls = 0

    def get(self, url):
        self.calls += 1
        self.remaining_failures -= 1
        if self.remaining_failures >= 0:
            raise TimeoutError("fail")
        return "success"


def retries_logged():
    return REGISTRY.get_sample_value("connection_retry_total") or 0


def test_no_problem_case():
    connection = FailedConnection(0)
    assert get_data(connection, "") == "success"
    assert connection.calls == 1


def test_successful_recovery():
    connection = FailedConnection(2)
    assert get_data(connection, "") == "success"
    assert connection.calls == 3


def test_exception_if_unable_to_recover():
    connection = FailedConnection(10)
    with pytest.raises(TimeoutError):
        get_data(connection, "")
    assert connection.calls == 3


def test_retries_are_logged():
    before = retries_logged()
    get_data(FailedConnection(2), "")
    assert retries_logged() - before == 2


def test_other_errors_are_not_retried():
    class BrokenConnection(Connection):
        def __init__(self):
            self.calls = 0

        def get(self, url):
            self.calls += 1
            raise PermissionError("403")

    connection = BrokenConnection()
    with pytest.raises(PermissionError):
        get_data(connection, "")
    assert connection.calls == 1
