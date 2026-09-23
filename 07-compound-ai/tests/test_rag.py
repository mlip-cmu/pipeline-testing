from unittest.mock import Mock

import pytest

from compound.prompts import PromptTemplate
from compound.rag import NO_ANSWER, TfidfRetriever, answer_question, load_documents
from fakes import ScriptedLLM

RETRIEVAL_CASES = [
    ("How much is the delivery fee?", "fees"),
    ("Is a refund possible if my order was late?", "refunds"),
    ("Do you deliver to Squirrel Hill?", "delivery-zones"),
    ("Until when do you deliver on Saturdays?", "opening-hours"),
    ("How old do couriers have to be?", "couriers"),
    ("I have a peanut allergy", "allergies"),
    pytest.param("My order was an hour late, do I get money back?", "refunds",
                 marks=pytest.mark.xfail(strict=True, reason="lexical retrieval misses paraphrases")),
    pytest.param("Until when can I order on Saturday night?", "opening-hours",
                 marks=pytest.mark.xfail(strict=True, reason="lexical retrieval misses paraphrases")),
]


@pytest.fixture(scope="module")
def retriever():
    return TfidfRetriever(load_documents())


@pytest.mark.parametrize("question, expected", RETRIEVAL_CASES)
def test_retrieval_finds_relevant_document(retriever, question, expected):
    assert expected in [d.name for d in retriever.retrieve(question, k=2)]


def test_retrieved_context_and_question_are_in_prompt(retriever):
    llm = ScriptedLLM("Orders of $20 or more have free delivery.")
    answer = answer_question("How much is the delivery fee?", retriever, llm)
    assert answer.text == "Orders of $20 or more have free delivery."
    assert "fees" in answer.sources
    assert "$2.99" in llm.prompts[0]
    assert "Question: How much is the delivery fee?" in llm.prompts[0]


def test_no_llm_call_without_relevant_documents(retriever):
    llm = Mock()
    answer = answer_question("What is the capital of Mongolia?", retriever, llm)
    assert answer.text == NO_ANSWER
    llm.assert_not_called()


def test_older_prompt_version_can_be_used(retriever):
    llm = ScriptedLLM("Free.")
    answer_question("delivery fee?", retriever, llm, PromptTemplate.load("answer_question", "v1"))
    assert llm.prompts[0].startswith("Answer the question using the context.")
