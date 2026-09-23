import pytest

from ensemble.background import BackgroundServer
from ensemble.model_server import create_app


@pytest.fixture(scope="module")
def model_apis():
    servers = {
        "M1": BackgroundServer(create_app("linear")),
        "M2": BackgroundServer(create_app("neighborhood")),
        "M3": BackgroundServer(create_app("conservative", delay_s=2.0)),
    }
    for server in servers.values():
        server.start()
    yield {name: f"{server.url}/predict" for name, server in servers.items()}
    for server in servers.values():
        server.stop()
