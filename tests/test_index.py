import pytest

from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from index.index import build_index


def test_index_success():
    documents = [
        Document(
            page_content="Grandma tells stories to the children.",
            metadata={
                "source": "book.pdf",
                "page": 1,
                "chunk_id": 0,
            },
        ),
        Document(
            page_content="The children listen to the story.",
            metadata={
                "source": "book.pdf",
                "page": 2,
                "chunk_id": 1,
            },
        ),
    ]

    embeddings = FakeEmbeddings(size=10)

    store = build_index(
        chunks=documents,
        embeddings=embeddings,
    )

    assert store.index.ntotal == 2


def test_index_failure():
    embeddings = FakeEmbeddings(size=10)

    with pytest.raises(
        ValueError,
        match="No chunks provided for indexing",
    ):
        build_index(
            chunks=[],
            embeddings=embeddings,
        )