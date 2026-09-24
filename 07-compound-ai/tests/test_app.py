"""System test for the user story "recap shows unread messages", with recorded model replies."""

from pathlib import Path

from fastapi.testclient import TestClient

from compound.app import create_app
from compound.replay import ReplayLLM

RECORDINGS = Path(__file__).parents[1] / "recordings"
MSG_1 = {"id": 1, "sender": "alice", "text": "Can you send me the slides?"}
MSG_2 = {"id": 2, "sender": "bob", "text": "The demo server is down, can you restart it?"}


def test_recap_lists_unread_only():
    llm = ReplayLLM(RECORDINGS / "recap.json")
    app = TestClient(create_app(llm))
    app.post("/send", json=MSG_1)
    app.post("/send", json=MSG_2)
    app.post("/read", json={"id": 1})
    r = app.get("/recap")
    assert r.status_code == 200
    assert r.json()["ids"] == [2]


def test_reading_unknown_message_is_an_error():
    app = TestClient(create_app(ReplayLLM(RECORDINGS / "recap.json")))
    assert app.post("/read", json={"id": 42}).status_code == 404
