from langchain_core.documents import Document

from evaluation.evaluation import (
    check_answer,
    check_citation,
)


def test_evaluation_success():
    answer = (
        "The author is Sudha Murty "
        "[data/book.pdf p3]."
    )

    document = Document(
        page_content="Sudha Murty is the author.",
        metadata={
            "source": "data/book.pdf",
            "page_label": "3",
            "chunk_id": 0,
        },
    )

    results = [
        (document, 0.80),
    ]

    answer_correct = check_answer(
        answer=answer,
        expected="Sudha Murty",
        should_refuse=False,
    )

    citation_correct = check_citation(
        answer,
        results,
    )

    assert answer_correct is True
    assert citation_correct is True


def test_evaluation_failure():
    answer = (
        "The author is Sudha Murty "
        "[data/book.pdf p99]."
    )

    document = Document(
        page_content="Sudha Murty is the author.",
        metadata={
            "source": "data/book.pdf",
            "page_label": "3",
            "chunk_id": 0,
        },
    )

    results = [
        (document, 0.80),
    ]

    citation_correct = check_citation(
        answer,
        results,
    )

    assert citation_correct is False


def test_refusal_accuracy():
    answer = "insufficient evidence"

    correct = check_answer(
        answer=answer,
        expected=None,
        should_refuse=True,
    )

    assert correct is True