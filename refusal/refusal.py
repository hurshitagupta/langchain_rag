import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


INDEX_PATH = "vector_store"
TOP_K = 4
SCORE_THRESHOLD = 0.05


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
    """Load the FAISS vector store."""

    embeddings = get_embeddings()

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_with_scores(store, question: str):
    """Retrieve documents together with relevance scores."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    results = store.similarity_search_with_relevance_scores(
        question,
        k=TOP_K,
    )

    return results


def has_sufficient_evidence(
    results,
    threshold: float = SCORE_THRESHOLD,
):
    """Check whether the best retrieved result passes the threshold."""

    if not results:
        return False

    top_score = results[0][1]

    return top_score >= threshold


def format_context(results):
    """Format retrieved documents with citations."""

    formatted_documents = []

    for document, score in results:
        source = document.metadata.get("source", "unknown")

        page = document.metadata.get("page_label")

        if page is None:
            page = document.metadata.get("page", "unknown")

        formatted_documents.append(
            f"[{source} p{page}] {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


def generate_answer(question: str, context: str, model):
    """Generate answer only from retrieved context."""

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the context below.

Do not use outside knowledge.

Every factual statement must include an inline citation
in the format [source pN].

Context:
{context}

Question:
{question}
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
    """Run RAG with score-based refusal."""

    store = load_index()

    results = retrieve_with_scores(
        store,
        question,
    )

    if not has_sufficient_evidence(results):
        return "insufficient evidence", results

    context = format_context(results)

    model = get_model()

    answer = generate_answer(
        question,
        context,
        model,
    )

    return answer, results


if __name__ == "__main__":
    questions = [
        "Who is the author of the book?",
        "What is the capital city of Brazil?",
    ]

    for question in questions:
        print("\nQUESTION:")
        print(question)

        answer, results = answer_question(question)

        print("\nRETRIEVAL SCORES:")

        for document, score in results:
            print(
                {
                    "score": round(score, 4),
                    "page": document.metadata.get("page_label"),
                    "chunk_id": document.metadata.get("chunk_id"),
                }
            )

        if results:
            print(
                f"\nTop score: {results[0][1]:.4f}"
            )

        print(
            f"Threshold: {SCORE_THRESHOLD}"
        )

        print("\nANSWER:")
        print(answer)

        print("-" * 50)