import pytest

from covid.features import build_feature_table
from covid.model import train
from covid.recordings import simulate_recordings


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, message):
        self.messages.append(message)


@pytest.fixture
def notifier():
    return FakeNotifier()


@pytest.fixture(scope="session")
def train_recordings():
    return simulate_recordings(1200, seed=1)


@pytest.fixture(scope="session")
def test_recordings():
    return simulate_recordings(1200, seed=2)


@pytest.fixture(scope="session")
def train_data(train_recordings):
    return build_feature_table(train_recordings)


@pytest.fixture(scope="session")
def test_data(test_recordings):
    return build_feature_table(test_recordings)


@pytest.fixture(scope="session")
def model(train_data):
    return train(train_data)
