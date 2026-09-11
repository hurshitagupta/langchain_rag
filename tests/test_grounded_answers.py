import pytest

from langchain_core.documents import Document

from grounded_answers.grounded_answers import (
    format_context,
    retrieve_documents,
)


class FakeStore:
    def similarity_search(self, question, k=4):
        return [
            Document(
                page_content="Sudha Murty is the author of the book.",
                metadata={
                    "source": "book.pdf",
                    "page": 2,
                    "page_label": "3",
                    "chunk_id": 0,
                },
            )
        ]


def test_grounded_retrieval_success():
    store = FakeStore()

    documents = retrieve_documents(store,"Who is the author?")

    context = format_context(documents)

    assert len(documents) == 1
    assert "Sudha Murty" in context
    assert "[book.pdf p3]" in context


def test_empty_question_failure():
    store = FakeStore()

    with pytest.raises(ValueError,match="Question cannot be empty"):
        retrieve_documents(store, "")