import pytest

from ingestion.ingestion import (
    clean_text,
    chunk_pages,
    calculate_statistics,
)

from langchain_core.documents import Document


def test_ingestion_success():
    pages = [
        Document(
            page_content=("LangChain retrieval augmented generation. " * 40),
            metadata={
                "source": "test.pdf",
                "page": page_number,
            },
        )
        for page_number in range(50)
    ]

    chunks = chunk_pages(pages)

    stats = calculate_statistics(pages, chunks)

    assert stats["pages_loaded"] == 50
    assert stats["chunks_created"] > 0

    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["source"] == "test.pdf"
    assert "page" in chunks[0].metadata


def test_clean_text_failure_case():
    messy_text = "LangChain      RAG\n\nretrieval     pipeline"

    cleaned = clean_text(messy_text)

    assert cleaned == "LangChain RAG retrieval pipeline"


def test_missing_pdf():
    from ingestion.ingestion import load_pdf

    with pytest.raises(FileNotFoundError):
        load_pdf("data/file_that_does_not_exist.pdf")