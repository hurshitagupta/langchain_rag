import pytest

from langchain_core.documents import Document

from refusal.refusal import (
    has_sufficient_evidence,
    format_context,
)


def test_sufficient_evidence_success():
    document = Document(
        page_content="Sudha Murty is the author.",
        metadata={
            "source": "book.pdf",
            "page_label": "3",
            "chunk_id": 0,
        },
    )

    results = [(document, 0.80)]

    assert has_sufficient_evidence(results,threshold=0.35) is True

    context = format_context(results)

    assert "[book.pdf p3]" in context


def test_weak_evidence_refusal():
    document = Document(
        page_content="Some unrelated text.",
        metadata={
            "source": "book.pdf",
            "page_label": "40",
            "chunk_id": 10,
        },
    )

    results = [(document, 0.20)]

    assert has_sufficient_evidence(results,threshold=0.35) is False


def test_no_results_refusal():
    assert has_sufficient_evidence([],threshold=0.35) is False