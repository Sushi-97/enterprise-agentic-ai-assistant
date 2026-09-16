from apps.data_service.infrastructure.connectors.local_documents import (
    load_documents,
)
from apps.data_service.infrastructure.indexing.chunking.markdown_section_chunker import (
    chunk_by_sections,
)
from apps.data_service.infrastructure.indexing.keyword.bm25_index import (
    BM25Index,
)
from tests.evaluation.retrieval_eval import TEST_CASES


def evaluate_keyword_retrieval(
    index: BM25Index,
    test_cases: list[dict],
    top_k: int = 3,
) -> None:
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_rank_sum = 0.0

    for test in test_cases:
        results = index.search(
            query=test["question"],
            top_k=top_k,
        )

        correct_rank = None

        for rank, (document, _) in enumerate(
            results,
            start=1,
        ):
            source_match = (
                document.metadata["source"]
                == test["expected_source"]
            )

            section_match = (
                test["expected_section"].lower()
                in document.page_content.lower()
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

        if correct_rank is not None:
            print(f"Correct chunk rank: {correct_rank}")
        else:
            print(
                f"Correct chunk not found in top {top_k}"
            )

    total = len(test_cases)

    print("\n" + "=" * 50)
    print("KEYWORD RETRIEVAL EVALUATION")
    print("=" * 50)
    print(f"Test cases:    {total}")
    print(f"Hit Rate@1:   {hit_at_1 / total:.2%}")
    print(f"Hit Rate@{top_k}:   {hit_at_k / total:.2%}")
    print(
        f"MRR:          "
        f"{reciprocal_rank_sum / total:.3f}"
    )


if __name__ == "__main__":
    documents = load_documents(
        "data/documents"
    )

    chunks = chunk_by_sections(documents)

    index = BM25Index(chunks)

    evaluate_keyword_retrieval(
        index=index,
        test_cases=TEST_CASES,
        top_k=3,
    )