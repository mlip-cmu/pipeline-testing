"""Evaluate prompt versions for the help center assistant against a small labeled question set."""

import argparse

from compound.llm import LiteLLM
from compound.prompts import PromptTemplate
from compound.rag import TfidfRetriever, answer_question, load_documents

EVAL_SET = [
    ("How much is delivery for a $15 order?", ["2.99"]),
    ("Is delivery free for a $25 order?", ["free"]),
    ("When are couriers paid?", ["tuesday"]),
    ("Do you deliver to Shadyside?", ["yes"]),
    ("How long until I see a refund on my card?", ["5 business days", "five business days"]),
    ("Can I order delivery at midnight on a Saturday?", ["yes", "01:00", "1 am", "1:00"]),
    ("Can you guarantee my food has no peanuts?", ["cannot", "can't", "not guarantee", "no guarantee"]),
    ("What is your delivery fee in Paris?", ["don't know", "support@quickbite"]),
]


def evaluate(llm, version: str) -> float:
    retriever = TfidfRetriever(load_documents())
    template = PromptTemplate.load("answer_question", version)
    correct = 0
    for question, expected in EVAL_SET:
        answer = answer_question(question, retriever, llm, template).text
        ok = any(e in answer.lower() for e in expected)
        correct += ok
        print(f"  [{'ok' if ok else 'FAIL'}] {question} -> {answer[:100]!r}")
    return correct / len(EVAL_SET)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=["v1", "v2"])
    parser.add_argument("--model", default=None, help="litellm model name, e.g. openai/gpt-4o-mini")
    args = parser.parse_args()
    llm = LiteLLM(args.model)
    results = {}
    for version in args.versions:
        print(f"Prompt answer_question_{version} with {llm.model}:")
        results[version] = evaluate(llm, version)
    for version, accuracy in results.items():
        print(f"{version}: {accuracy:.0%} correct")


if __name__ == "__main__":
    main()
