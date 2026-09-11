from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/book.pdf"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

MIN_PAGES = 50


def clean_text(text: str) -> str:
    """Remove unnecessary whitespace from extracted PDF text."""

    return " ".join(text.split())


def load_pdf(pdf_path: str):
    """Load PDF pages."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    if len(pages) < MIN_PAGES:
        raise ValueError(
            f"PDF must contain at least {MIN_PAGES} pages. "
            f"Found only {len(pages)} pages."
        )

    return pages


def clean_pages(pages):
    """Clean text while keeping page metadata."""

    cleaned_pages = []

    for page in pages:
        page.page_content = clean_text(page.page_content)
        cleaned_pages.append(page)

    return cleaned_pages


def chunk_pages(pages):
    """Split pages into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


def calculate_statistics(pages, chunks):
    """Calculate simple ingestion statistics."""

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]

    average_chunk_size = (
        sum(chunk_lengths) / len(chunk_lengths)
        if chunk_lengths
        else 0
    )

    return {
        "pages_loaded": len(pages),
        "chunks_created": len(chunks),
        "average_chunk_size": round(average_chunk_size, 2),
        "smallest_chunk": min(chunk_lengths) if chunk_lengths else 0,
        "largest_chunk": max(chunk_lengths) if chunk_lengths else 0,
    }


def ingest_pdf(pdf_path: str):
    """Run complete ingestion pipeline."""

    pages = load_pdf(pdf_path)

    cleaned_pages = clean_pages(pages)

    chunks = chunk_pages(cleaned_pages)

    stats = calculate_statistics(cleaned_pages, chunks)

    return chunks, stats


if __name__ == "__main__":
    chunks, stats = ingest_pdf(PDF_PATH)

    print("RAG INGESTION COMPLETE")
    print("----------------------")

    print(f"Pages loaded: {stats['pages_loaded']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Average chunk size: {stats['average_chunk_size']}")
    print(f"Smallest chunk: {stats['smallest_chunk']}")
    print(f"Largest chunk: {stats['largest_chunk']}")

    print("\nSample chunk metadata:")
    print(chunks[0].metadata)

    print("\nSample chunk:")
    print(chunks[0].page_content[:300])