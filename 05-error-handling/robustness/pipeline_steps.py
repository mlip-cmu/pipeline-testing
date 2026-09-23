"""Pipeline steps that load data and deploy models, with explicit error handling."""

import hashlib
import io
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

import httpx
import pandas as pd
from prometheus_client import Counter

log = logging.getLogger(__name__)
invalid_rows_counter = Counter("invalid_training_rows_total", "Training rows removed by validation")
REQUIRED_COLUMNS = {"datetime", "weather", "temp", "delivery_count"}


class MissingDataError(Exception):
    pass


class StaleDataError(Exception):
    pass


class InvalidDataError(Exception):
    pass


class DeploymentError(Exception):
    pass


@dataclass
class DataBatch:
    data: pd.DataFrame
    updated_at: datetime


class DataSource(Protocol):
    def fetch(self) -> DataBatch: ...


class HttpCsvSource:
    def __init__(self, url: str):
        self.url = url

    def fetch(self) -> DataBatch:
        response = httpx.get(self.url, timeout=30)
        response.raise_for_status()
        modified = response.headers.get("last-modified")
        updated_at = pd.Timestamp(modified).to_pydatetime() if modified else datetime.now(timezone.utc)
        return DataBatch(pd.read_csv(io.StringIO(response.text)), updated_at)


def validate(data: pd.DataFrame, max_invalid_fraction: float = 0.05) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise InvalidDataError(f"Missing columns: {sorted(missing)}")
    invalid = data["delivery_count"].isna() | (data["delivery_count"] < 0) | ~data["weather"].isin([1, 2, 3, 4])
    if invalid.mean() > max_invalid_fraction:
        raise InvalidDataError(f"{invalid.mean():.0%} of rows are invalid")
    if invalid.any():
        log.warning("Removed %d invalid rows", invalid.sum())
        invalid_rows_counter.inc(int(invalid.sum()))
    return data[~invalid]


def load_training_data(source: DataSource, max_age: timedelta = timedelta(days=2),
                       now: datetime | None = None) -> pd.DataFrame:
    try:
        batch = source.fetch()
    except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
        raise MissingDataError(f"Data source unavailable: {e}") from e
    if batch.data.empty:
        raise MissingDataError("Data source returned no rows")
    age = (now or datetime.now(timezone.utc)) - batch.updated_at
    if age > max_age:
        raise StaleDataError(f"Data was last updated {age} ago")
    return validate(batch.data)


class ModelStore(Protocol):
    def upload(self, name: str, payload: bytes) -> None: ...
    def checksum(self, name: str) -> str | None: ...


class Notifier(Protocol):
    def send_notification(self, message: str) -> None: ...


def deploy(model, name: str, store: ModelStore, notifier: Notifier) -> str:
    payload = pickle.dumps(model)
    expected = hashlib.sha256(payload).hexdigest()
    try:
        store.upload(name, payload)
        actual = store.checksum(name)
        if actual != expected:
            raise DeploymentError(f"Checksum mismatch after upload: {actual} != {expected}")
    except Exception as e:
        notifier.send_notification(f"Deployment of {name} failed: {e}")
        if isinstance(e, DeploymentError):
            raise
        raise DeploymentError(str(e)) from e
    log.info("Deployed %s (%s)", name, expected[:8])
    return expected
