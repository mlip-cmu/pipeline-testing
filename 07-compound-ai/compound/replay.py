"""Record real model replies once, replay them in tests."""

import json
from pathlib import Path

from compound.llm import LLM


class MissingRecording(KeyError):
    pass


class ReplayLLM:
    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text())
        self.model = data["model"]
        self.replies = {r["prompt"]: r["reply"] for r in data["recordings"]}

    def __call__(self, prompt: str) -> str:
        try:
            return self.replies[prompt]
        except KeyError:
            raise MissingRecording(f"No recording for this prompt (the prompt changed?); re-record with "
                                   f"`python -m compound.record`:\n{prompt[:200]}") from None


class RecordingLLM:
    def __init__(self, llm: LLM, path: str | Path, model: str):
        self.llm = llm
        self.path = Path(path)
        self.model = model
        self.recordings: list[dict] = []

    def __call__(self, prompt: str) -> str:
        reply = self.llm(prompt)
        self.recordings.append({"prompt": prompt, "reply": reply})
        return reply

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"model": self.model, "recordings": self.recordings}, indent=2) + "\n")
