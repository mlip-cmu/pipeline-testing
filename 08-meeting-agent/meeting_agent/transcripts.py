"""Meeting transcripts: '[00:03:12] Ann: I finished the login page.'"""

import re
from dataclasses import dataclass
from pathlib import Path

LINE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.*)$")


@dataclass
class Utterance:
    time: str
    speaker: str
    text: str


def load_transcript(path: str | Path) -> list[Utterance]:
    utterances = []
    for line in Path(path).read_text().splitlines():
        match = LINE.match(line.strip())
        if match:
            utterances.append(Utterance(*match.groups()))
        elif utterances and line.strip():
            utterances[-1].text += " " + line.strip()
    return utterances


def transcribe(audio_path: str | Path) -> list[Utterance]:
    raise NotImplementedError("connect a speech-to-text service here")


def format_transcript(utterances: list[Utterance], max_chars: int = 12_000) -> str:
    text = "\n".join(f"[{u.time}] {u.speaker}: {u.text}" for u in utterances)
    return text[-max_chars:]
