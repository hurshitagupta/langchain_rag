import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from ingestion.ingestion import ingest_pdf


load_dotenv()


PDF_PATH = "data/book.pdf"
INDEX_PATH = "vector_store"


def get_embeddings():
    """Create the embedding model."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("BASE_URL")
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "openai/text-embedding-3-small",
    )

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    return OpenAIEmbeddings(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        request_timeout=20,
        max_retries=3,
    )


def build_index(chunks, embeddings):
    """Create a FAISS vector store from document chunks."""

    if not chunks:
        raise ValueError("No chunks provided for indexing")

    store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return store


def save_index(store, path: str):
    """Save FAISS index locally."""

    store.save_local(path)


def create_index(pdf_path: str):
    """Run ingestion and indexing."""

    chunks, stats = ingest_pdf(pdf_path)

    print(f"Chunks received from ingestion: {len(chunks)}")

    embeddings = get_embeddings()

    store = build_index(
        chunks=chunks,
        embeddings=embeddings,
    )

    save_index(store, INDEX_PATH)

    return store, chunks, stats


if __name__ == "__main__":
    store, chunks, stats = create_index(PDF_PATH)

    print("\nRAG INDEXING COMPLETE")
    print("---------------------")

    print(f"Pages processed: {stats['pages_loaded']}")
    print(f"Chunks indexed: {store.index.ntotal}")
    print(f"Vector store saved to: {INDEX_PATH}")

    print("\nSample indexed metadata:")
    print(chunks[0].metadata)