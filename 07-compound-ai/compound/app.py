"""A small chat service with a recap endpoint. Run: uv run uvicorn compound.app:app"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from compound.llm import LLM, LiteLLM
from compound.recap import Message, summarize_unread


class NewMessage(BaseModel):
    id: int
    sender: str
    text: str


class ReadReceipt(BaseModel):
    id: int


def create_app(llm: LLM) -> FastAPI:
    app = FastAPI(title="chat with recap")
    messages: dict[int, Message] = {}

    @app.post("/send")
    def send(msg: NewMessage):
        messages[msg.id] = Message(msg.id, msg.sender, msg.text)
        return {"id": msg.id}

    @app.post("/read")
    def read(receipt: ReadReceipt):
        if receipt.id not in messages:
            raise HTTPException(404, f"no message {receipt.id}")
        messages[receipt.id].read = True
        return {"id": receipt.id}

    @app.get("/recap")
    def recap():
        result = summarize_unread(list(messages.values()), llm)
        return {"summary": result.summary, "ids": result.message_ids}

    return app


def __getattr__(name):
    if name == "app":
        return create_app(LiteLLM())
    raise AttributeError(name)
