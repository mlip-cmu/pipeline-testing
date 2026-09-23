"""Start three model services (one slow) and query the ensemble."""

import asyncio

from ensemble.background import BackgroundServer
from ensemble.model_server import create_app
from ensemble.predict import TooManyModelsFailed, predict_price

if __name__ == "__main__":
    servers = [BackgroundServer(create_app(name, delay)).start()
               for name, delay in [("linear", 0), ("neighborhood", 0), ("conservative", 2.0)]]
    urls = [f"{s.url}/predict" for s in servers]
    house = {"sqft": 1500, "bedrooms": 3}
    for timeout_ms in (500, 3000):
        print(f"timeout {timeout_ms} ms:", asyncio.run(predict_price(house, urls, timeout_ms)))
    try:
        asyncio.run(predict_price(house, [urls[0], urls[2]], 500))
    except TooManyModelsFailed as e:
        print("timeout 500 ms, one fast and one slow model:", e)
    for s in servers:
        s.stop()
