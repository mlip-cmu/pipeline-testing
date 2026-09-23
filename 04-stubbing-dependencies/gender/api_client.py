"""Client for the genderize.io web API, which guesses gender from a first name."""

import httpx

COUNTRIES = {"italy": "IT", "france": "FR", "germany": "DE", "usa": "US", "uk": "GB", "spain": "ES"}
US_STATES = {"AL", "AK", "AZ", "CA", "CO", "FL", "GA", "IL", "MA", "MI", "NY", "OH", "PA", "TX", "WA"}


def country_code(location: str) -> str | None:
    region = location.split(",")[-1].strip()
    if region.upper() in US_STATES:
        return "US"
    return COUNTRIES.get(region.lower())


class GenderApiClient:
    def __init__(self, base_url: str = "https://api.genderize.io", min_probability: float = 0.8,
                 http: httpx.Client | None = None):
        self.base_url = base_url
        self.min_probability = min_probability
        self.http = http or httpx.Client(timeout=5)

    def predict(self, firstname: str, lastname: str, location: str) -> str | None:
        params = {"name": firstname}
        if country := country_code(location):
            params["country_id"] = country
        response = self.http.get(self.base_url, params=params)
        response.raise_for_status()
        result = response.json()
        if result.get("gender") is None or result.get("probability", 0) < self.min_probability:
            return None
        return {"male": "M", "female": "F"}[result["gender"]]


gender_api_client = GenderApiClient()
