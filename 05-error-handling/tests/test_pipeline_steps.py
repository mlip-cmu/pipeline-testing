import hashlib
import logging
import pickle
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from robustness.pipeline_steps import (
    DataBatch,
    DeploymentError,
    InvalidDataError,
    MissingDataError,
    StaleDataError,
    deploy,
    load_training_data,
    validate,
)

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def rows(n=100, **overrides):
    data = pd.DataFrame({"datetime": ["2026-08-31 12:00"] * n, "weather": [1] * n, "temp": [20.0] * n,
                         "delivery_count": [50.0] * n})
    for column, values in overrides.items():
        data[column] = list(values) + data[column].tolist()[len(values):]
    return data


class StubSource:
    def __init__(self, data=None, updated_at=NOW, error=None):
        self.data = rows() if data is None else data
        self.updated_at = updated_at
        self.error = error

    def fetch(self):
        if self.error:
            raise self.error
        return DataBatch(self.data, self.updated_at)


class StubNotifier:
    def __init__(self):
        self.messages = []

    def send_notification(self, message):
        self.messages.append(message)


class InMemoryStore:
    def __init__(self, fail_upload=False, corrupt=False):
        self.files = {}
        self.fail_upload = fail_upload
        self.corrupt = corrupt

    def upload(self, name, payload):
        if self.fail_upload:
            raise httpx.ConnectError("storage unreachable")
        self.files[name] = payload[:-1] if self.corrupt else payload

    def checksum(self, name):
        return hashlib.sha256(self.files[name]).hexdigest() if name in self.files else None


def test_valid_data_is_loaded():
    assert len(load_training_data(StubSource(), now=NOW)) == 100


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), TimeoutError("slow"), ConnectionError()])
def test_unavailable_source_raises_missing_data(error):
    with pytest.raises(MissingDataError):
        load_training_data(StubSource(error=error), now=NOW)


def test_empty_data_raises_missing_data():
    with pytest.raises(MissingDataError):
        load_training_data(StubSource(data=rows(0)), now=NOW)


def test_missing_data_update_raises_stale_data():
    with pytest.raises(StaleDataError):
        load_training_data(StubSource(updated_at=NOW - timedelta(days=5)), now=NOW)


def test_few_invalid_rows_are_repaired_and_logged(caplog):
    before = REGISTRY.get_sample_value("invalid_training_rows_total") or 0
    with caplog.at_level(logging.WARNING):
        data = validate(rows(delivery_count=[-1.0, None], weather=[1, 1, 7]))
    assert len(data) == 97
    assert "Removed 3 invalid rows" in caplog.text
    assert REGISTRY.get_sample_value("invalid_training_rows_total") - before == 3


def test_many_invalid_rows_are_rejected():
    with pytest.raises(InvalidDataError, match="20%"):
        validate(rows(delivery_count=[-1.0] * 20))


def test_missing_columns_are_rejected():
    with pytest.raises(InvalidDataError, match="temp"):
        validate(rows().drop(columns=["temp"]))


def test_successful_deployment():
    store, notifier = InMemoryStore(), StubNotifier()
    checksum = deploy({"weights": [1, 2]}, "model-v2", store, notifier)
    assert pickle.loads(store.files["model-v2"]) == {"weights": [1, 2]}
    assert checksum == store.checksum("model-v2")
    assert notifier.messages == []


@pytest.mark.parametrize("store", [InMemoryStore(fail_upload=True), InMemoryStore(corrupt=True)])
def test_failing_deployment_is_reported(store):
    notifier = StubNotifier()
    with pytest.raises(DeploymentError):
        deploy({"weights": [1, 2]}, "model-v2", store, notifier)
    assert "model-v2 failed" in notifier.messages[0]
