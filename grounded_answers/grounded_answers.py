import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


INDEX_PATH = "vector_store"
TOP_K = 4


def get_embeddings():
    """Create the same embedding model used during indexing."""

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


def get_model():
    """Create the chat model."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("BASE_URL")
    model_name = os.getenv("MODEL_NAME")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    if not model_name:
        raise ValueError("MODEL_NAME is missing")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=20,
        max_retries=3,
        max_tokens=300,
    )


def load_index():
    """Load the FAISS index created in Task 2."""

    embeddings = get_embeddings()

    store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return store


def retrieve_documents(store, question: str):
    """Retrieve the most relevant chunks."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    documents = store.similarity_search(
        question,
        k=TOP_K,
    )

    return documents


def format_context(documents):
    """Format retrieved documents with citation information."""

    formatted_documents = []

    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page_label")

        if page is None:
            page = document.metadata.get("page", "unknown")

        formatted_documents.append(
            f"[{source} p{page}] {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


def generate_answer(question: str, context: str, model):
    """Generate an answer using only retrieved context."""

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the context below.

Do not use outside knowledge.

Every factual statement in the answer must include an inline
citation in the format [source pN].

If the context does not contain enough information to answer,
reply exactly:

insufficient evidence

Context:{context}

Question:{question}
"""
    )

    chain = prompt | model | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer.strip()


def answer_question(question: str):
    """Run the grounded RAG pipeline."""

    store = load_index()

    documents = retrieve_documents(
        store,
        question,
    )

    context = format_context(documents)

    model = get_model()

    answer = generate_answer(question,context,model)

    return answer, documents


if __name__ == "__main__":
    question = "Who is the author of the book?"

    answer, documents = answer_question(question)

    print("QUESTION:")
    print(question)

    print("\nRETRIEVED CHUNKS:")
    print(len(documents))

    print("\nRETRIEVED METADATA:")

    for document in documents:
        print(
            {
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "page_label": document.metadata.get("page_label"),
                "chunk_id": document.metadata.get("chunk_id"),
            }
        )

    print("\nGROUNDED ANSWER:")
    print(answer)