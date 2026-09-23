"""A house price model behind a REST API. Configure with MODEL_NAME and MODEL_DELAY_S."""

import asyncio
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS = {
    "linear": {"base": 50_000, "per_sqft": 180, "per_bedroom": 10_000},
    "neighborhood": {"base": 80_000, "per_sqft": 160, "per_bedroom": 15_000},
    "conservative": {"base": 40_000, "per_sqft": 150, "per_bedroom": 5_000},
}


class House(BaseModel):
    sqft: float = Field(gt=0)
    bedrooms: int = Field(ge=0)


class PricePrediction(BaseModel):
    model: str
    price: float


def create_app(model_name: str = "linear", delay_s: float = 0.0) -> FastAPI:
    coefficients = MODELS[model_name]
    app = FastAPI(title=f"price model: {model_name}")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=PricePrediction)
    async def predict(house: House):
        await asyncio.sleep(delay_s)
        price = coefficients["base"] + coefficients["per_sqft"] * house.sqft + coefficients["per_bedroom"] * house.bedrooms
        return PricePrediction(model=model_name, price=price)

    return app


app = create_app(os.environ.get("MODEL_NAME", "linear"), float(os.environ.get("MODEL_DELAY_S", "0")))
