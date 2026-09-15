import sys
from pathlib import Path

# Add project root to Python path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "rag"))

from retriever import SemanticRetriever


TEST_CASES = [
    {
        "question": "How many annual leave days can I carry forward?",
        "expected_source": "leave_policy.md",
        "expected_section": "Carry Forward",
    },
    {
        "question": "How many paid sick leave days do employees receive?",
        "expected_source": "leave_policy.md",
        "expected_section": "Sick Leave",
    },
    {
        "question": "How far in advance should I request more than five days of leave?",
        "expected_source": "leave_policy.md",
        "expected_section": "Leave Request Process",
    },
    {
        "question": "What is the daily meal allowance during business travel?",
        "expected_source": "travel_policy.md",
        "expected_section": "Meals",
    },
    {
        "question": "Can I book business class for a flight longer than ten hours?",
        "expected_source": "travel_policy.md",
        "expected_section": "Air Travel",
    },
    {
        "question": "How soon after a trip must I submit my travel expenses?",
        "expected_source": "travel_policy.md",
        "expected_section": "Expense Submission",
    },
    {
        "question": "When do I need a receipt for an expense?",
        "expected_source": "expense_policy.md",
        "expected_section": "Receipt Requirements",
    },
    {
        "question": "Who needs to approve an expense report above $500?",
        "expected_source": "expense_policy.md",
        "expected_section": "Approval Process",
    },
    {
        "question": "Can I expense a new software subscription?",
        "expected_source": "expense_policy.md",
        "expected_section": "Software and Subscriptions",
    },
]


def evaluate_retriever(retriever, test_cases, top_k=3):
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_rank_sum = 0

    for test in test_cases:
        results = retriever.search(test["question"], top_k=top_k)

        correct_rank = None

        for rank, result in enumerate(results, start=1):
            source_match = (
                result["metadata"]["source"]
                == test["expected_source"]
            )

            section_match = (
                test["expected_section"].lower()
                in result["content"].lower()
            )

            if source_match and section_match:
                correct_rank = rank
                break

        if correct_rank == 1:
            hit_at_1 += 1

        if correct_rank is not None:
            hit_at_k += 1
            reciprocal_rank_sum += 1 / correct_rank

        print(f"\nQuestion: {test['question']}")

        if correct_rank:
            print(f"Correct chunk rank: {correct_rank}")
        else:
            print(f"Correct chunk not found in top {top_k}")

    total = len(test_cases)

    hit_rate_1 = hit_at_1 / total
    hit_rate_k = hit_at_k / total
    mrr = reciprocal_rank_sum / total

    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION")
    print("=" * 50)

    print(f"Test cases:   {total}")
    print(f"Hit Rate@1:  {hit_rate_1:.2%}")
    print(f"Hit Rate@{top_k}:  {hit_rate_k:.2%}")
    print(f"MRR:         {mrr:.3f}")


if __name__ == "__main__":
    retriever = SemanticRetriever("data/documents")

    evaluate_retriever(
        retriever=retriever,
        test_cases=TEST_CASES,
        top_k=3,
    )