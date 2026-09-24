"""Record model replies for the replay tests: uv run python -m compound.record [--model MODEL]"""

import argparse
from pathlib import Path

from compound.llm import LiteLLM
from compound.recap import Message, SAMPLE_MESSAGES, summarize_unread
from compound.replay import RecordingLLM

RECORDINGS = Path(__file__).parent.parent / "recordings" / "recap.json"
APP_MESSAGES = [Message(1, "alice", "Can you send me the slides?", read=True),
                Message(2, "bob", "The demo server is down, can you restart it?")]

HAND_WRITTEN = [
    '{"summary": "Standup moves to 10am tomorrow, and Alice asks you to review the deployment checklist '
    'before Friday.", "message_ids": [3, 7]}',
    '{"summary": "Bob says the demo server is down and asks you to restart it.", "message_ids": [2]}',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="litellm model name (default: LLM_MODEL)")
    parser.add_argument("--hand-written", action="store_true", help="store hand-written replies instead")
    args = parser.parse_args()
    if args.hand_written:
        replies = iter(HAND_WRITTEN)
        llm, model = (lambda prompt: next(replies)), "hand-written example replies"
    else:
        llm = LiteLLM(args.model)
        model = llm.model
    recorder = RecordingLLM(llm, RECORDINGS, model)
    for messages in (SAMPLE_MESSAGES, APP_MESSAGES):
        print(summarize_unread(messages, recorder))
    recorder.save()
    print(f"Saved {len(recorder.recordings)} recordings to {RECORDINGS}")


if __name__ == "__main__":
    main()
