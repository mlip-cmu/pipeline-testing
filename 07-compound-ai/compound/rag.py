"""Retrieval-augmented generation over the QuickBite help center documents."""

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from compound.llm import LLM
from compound.prompts import PromptTemplate, screen_user_input

DOCS_DIR = Path(__file__).parent.parent / "docs"
NO_ANSWER = "I don't know. Please contact support@quickbite.example."


@dataclass
class Document:
    name: str
    text: str


@dataclass
class Answer:
    text: str
    sources: list[str]


def load_documents(directory: Path = DOCS_DIR) -> list[Document]:
    return [Document(p.stem, p.read_text()) for p in sorted(directory.glob("*.md"))]


class TfidfRetriever:
    def __init__(self, documents: list[Document], min_score: float = 0.1):
        self.documents = documents
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(d.text for d in documents)

    def retrieve(self, query: str, k: int = 2) -> list[Document]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(zip(scores, self.documents), key=lambda pair: -pair[0])
        return [doc for score, doc in ranked[:k] if score >= self.min_score]


def answer_question(question: str, retriever: TfidfRetriever, llm: LLM,
                    template: PromptTemplate | None = None) -> Answer:
    question = screen_user_input(question)
    documents = retriever.retrieve(question)
    if not documents:
        return Answer(NO_ANSWER, [])
    template = template or PromptTemplate.load("answer_question", "v2")
    context = "\n\n".join(f"[{d.name}]\n{d.text.strip()}" for d in documents)
    return Answer(llm(template.render(context=context, question=question)).strip(), [d.name for d in documents])


if __name__ == "__main__":
    import sys

    from compound.llm import LiteLLM

    answer = answer_question(" ".join(sys.argv[1:]) or "How much is delivery?", TfidfRetriever(load_documents()), LiteLLM())
    print(f"{answer.text}\n(sources: {', '.join(answer.sources)})")
