import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LLM_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_LLM_TESTS=1 (and LLM_MODEL) to run tests against a real LLM")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
