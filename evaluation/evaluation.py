import re
import time
import unicodedata

from refusal.refusal import answer_question


EVALUATION_SET = [
    # Questions expected to be answerable from the book
    {
        "question": "Who is the author of the book?",
        "expected": "Sudha Murty",
        "should_refuse": False,
    },
    {
        "question": "What is the title of the book?",
        "expected": "Grandma's Bag of Stories",
        "should_refuse": False,
    },
    {
        "question": "Who illustrated the book?",
        "expected": "Priya Kuriyan",
        "should_refuse": False,
    },
    {
        "question": "In which year was the book first published?",
        "expected": "2012",
        "should_refuse": False,
    },
    {
        "question": "In which year was the digital edition published?",
        "expected": "2015",
        "should_refuse": False,
    },
    {
        "question": "Which company published the book?",
        "expected": "Penguin",
        "should_refuse": False,
    },
    {
        "question": "Who does Sudha Murty thank as her editor?",
        "expected": "Sudeshna Shome Ghosh",
        "should_refuse": False,
    },
    {
        "question": "Which city is mentioned with Sudha Murty's name in the introduction?",
        "expected": "Bangalore",
        "should_refuse": False,
    },
    {
        "question": "Who owns the text copyright?",
        "expected": "Sudha Murty",
        "should_refuse": False,
    },
    {
        "question": "Who owns the illustration copyright?",
        "expected": "Priya Kuriyan",
        "should_refuse": False,
    },

    # Questions intentionally outside the book
    {
        "question": "What is the capital city of Brazil?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the chemical symbol for gold?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "Who created the Python programming language?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the largest planet in the solar system?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the capital of France?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What currency is used in Japan?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "Who invented the telephone?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the tallest mountain in the world?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the boiling point of water?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the chemical formula of water?",
        "expected": None,
        "should_refuse": True,
    },
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Convert unusual spaces into normal spaces
    text = " ".join(text.split())

    # Normalize curly apostrophes
    text = text.replace("’", "'")

    return text.lower()


def check_answer(
    answer: str,
    expected: str | None,
    should_refuse: bool,
) -> bool:

    normalized_answer = normalize_text(answer)

    if should_refuse:
        return normalized_answer == "insufficient evidence"

    if not expected:
        return False

    normalized_expected = normalize_text(expected)

    return normalized_expected in normalized_answer


def extract_citations(answer: str):
    pattern = r"[\[【]\s*([^\]】]+?)\s+p(\d+)\s*[\]】]"

    citations = re.findall(pattern, answer)

    return [
        (source.strip(), page.strip())
        for source, page in citations
    ]


def check_citation(answer: str, results) -> bool:
    """Verify that an answer citation matches retrieved metadata."""

    citations = extract_citations(answer)

    if not citations:
        return False

    valid_citations = set()

    for document, _score in results:
        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get("page_label")

        if page is None:
            page = document.metadata.get(
                "page",
                "unknown",
            )

        valid_citations.add(
            (str(source), str(page))
        )

    for citation in citations:
        if citation in valid_citations:
            return True

    return False


def evaluate():
    """Run the complete 20-question evaluation."""

    total_questions = len(EVALUATION_SET)

    correct_answers = 0
    citation_checks = 0
    correct_citations = 0

    expected_refusals = 0
    correct_refusals = 0

    total_latency = 0

    print("RAG EVALUATION")
    print("==============")

    for number, case in enumerate(
        EVALUATION_SET,
        start=1,
    ):
        question = case["question"]
        expected = case["expected"]
        should_refuse = case["should_refuse"]

        start_time = time.perf_counter()

        answer, results = answer_question(question)

        latency = time.perf_counter() - start_time

        total_latency += latency

        answer_correct = check_answer(
            answer,
            expected,
            should_refuse,
        )

        if answer_correct:
            correct_answers += 1

        if should_refuse:
            expected_refusals += 1

            if answer.strip().lower() == "insufficient evidence":
                correct_refusals += 1

            citation_correct = None

        else:
            citation_checks += 1

            citation_correct = check_citation(
                answer,
                results,
            )

            if citation_correct:
                correct_citations += 1

        print(f"\nCase {number:02d}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")
        print(f"Should refuse: {should_refuse}")
        print(f"Answer: {answer}")
        print(f"Answer correct: {answer_correct}")

        if citation_correct is not None:
            print(
                f"Citation correct: {citation_correct}"
            )
        else:
            print("Citation correct: N/A (refusal expected)")

        print(f"Latency: {latency:.2f}s")

    answer_accuracy = (
        correct_answers / total_questions
    ) * 100

    citation_accuracy = (
        correct_citations / citation_checks
    ) * 100 if citation_checks else 0

    refusal_accuracy = (
        correct_refusals / expected_refusals
    ) * 100 if expected_refusals else 0

    average_latency = (
        total_latency / total_questions
    )

    print("\n\nEVALUATION SUMMARY")
    print("==================")

    print(
        f"Questions evaluated: {total_questions}"
    )

    print(
        f"Correct answers: "
        f"{correct_answers}/{total_questions}"
    )

    print(
        f"Answer accuracy: "
        f"{answer_accuracy:.2f}%"
    )

    print(
        f"Correct citations: "
        f"{correct_citations}/{citation_checks}"
    )

    print(
        f"Citation correctness: "
        f"{citation_accuracy:.2f}%"
    )

    print(
        f"Correct refusals: "
        f"{correct_refusals}/{expected_refusals}"
    )

    print(
        f"Refusal accuracy: "
        f"{refusal_accuracy:.2f}%"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f}s"
    )

    return {
        "total_questions": total_questions,
        "answer_accuracy": answer_accuracy,
        "citation_correctness": citation_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "average_latency": average_latency,
    }


if __name__ == "__main__":
    evaluate()