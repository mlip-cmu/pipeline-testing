import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption("--impl", choices=["fixed", "buggy"], default="fixed",
                     help="which implementation in functions-and-tests/wrangling to test")


@pytest.fixture
def w(request):
    return importlib.import_module(f"wrangling.{request.config.getoption('--impl')}")
