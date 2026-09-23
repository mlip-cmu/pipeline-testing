"""Test the API client itself by stubbing the HTTP layer (no network access)."""

import httpx
import pytest

from gender.api_client import GenderApiClient, country_code


def client_with(handler):
    return GenderApiClient(http=httpx.Client(transport=httpx.MockTransport(handler)))


def respond(gender, probability):
    return lambda request: httpx.Response(200, json={"name": "x", "gender": gender, "probability": probability})


@pytest.mark.parametrize("location, expected", [
    ("Pittsburgh, PA", "US"), ("Paris, PA ", "US"), ("Rome, Italy", "IT"), ("Atlantis", None),
])
def test_country_code(location, expected):
    assert country_code(location) == expected


def test_request_contains_name_and_country():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"gender": "female", "probability": 0.98})

    assert client_with(handler).predict("Andrea", "Rossi", "Rome, Italy") == "F"
    assert requests[0].url.params["name"] == "Andrea"
    assert requests[0].url.params["country_id"] == "IT"


@pytest.mark.parametrize("gender, probability, expected", [
    ("male", 0.99, "M"), ("female", 0.9, "F"), ("male", 0.55, None), (None, 0.0, None),
])
def test_response_mapping(gender, probability, expected):
    assert client_with(respond(gender, probability)).predict("Kim", "Lee", "Seattle, WA") == expected


def test_server_error_raises():
    client = client_with(lambda request: httpx.Response(429, json={"error": "Request limit reached"}))
    with pytest.raises(httpx.HTTPStatusError):
        client.predict("John", "Doe", "Pittsburgh, PA")
