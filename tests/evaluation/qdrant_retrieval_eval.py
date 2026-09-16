from qdrant_client import QdrantClient

from tests.evaluation.retrieval_eval import TEST_CASES

from apps.data_service.domain.models.retrieval import (
    RetrievalMode,
    RetrievalRequest,
)
from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.embeddings.local_sparse_embedding import (
    LocalSparseEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.qdrant_indexer import (
    QdrantDocumentIndexer,
)
from apps.data_service.infrastructure.retrieval.backends.qdrant_search_backend import (
    QdrantSearchBackend,
)


def is_relevant(result, test_case):
    return (
        result.source == test_case["expected_source"]
        and test_case["expected_section"].lower()
        in result.content.lower()
    )


def evaluate_mode(backend, mode):
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_rank_sum = 0.0

    for test_case in TEST_CASES:
        request = RetrievalRequest(
            query=test_case["question"],
            source="local_documents",
            top_k=3,
            retrieval_mode=mode,
        )

        if mode == RetrievalMode.SEMANTIC:
            results = backend.search_semantic(request)
        elif mode == RetrievalMode.KEYWORD:
            results = backend.search_keyword(request)
        else:
            results = backend.search_hybrid(request)

        relevant_rank = None

        for rank, result in enumerate(results, start=1):
            if is_relevant(result, test_case):
                relevant_rank = rank
                break

        if relevant_rank == 1:
            hits_at_1 += 1

        if relevant_rank is not None and relevant_rank <= 3:
            hits_at_3 += 1
            reciprocal_rank_sum += 1 / relevant_rank

    total = len(TEST_CASES)

    return {
        "hit_at_1": hits_at_1 / total,
        "hit_at_3": hits_at_3 / total,
        "mrr": reciprocal_rank_sum / total,
    }


def main():
    client = QdrantClient(path="data/qdrant")

    dense = LocalEmbeddingModel()
    sparse = LocalSparseEmbeddingModel()

    QdrantDocumentIndexer(
        client=client,
        collection_name="acmetech_policies",
        documents_directory="data/documents",
        embedding_model=dense,
        sparse_embedding_model=sparse,
    ).build()

    backend = QdrantSearchBackend(
        client=client,
        collection_name="acmetech_policies",
        embedding_model=dense,
        sparse_embedding_model=sparse,
    )

    print("=" * 65)
    print("QDRANT RETRIEVAL EVALUATION")
    print("=" * 65)

    for mode in RetrievalMode:
        metrics = evaluate_mode(backend, mode)

        print(f"\n{mode.value.upper()}")
        print(f"Hit Rate@1: {metrics['hit_at_1']:.2%}")
        print(f"Hit Rate@3: {metrics['hit_at_3']:.2%}")
        print(f"MRR:        {metrics['mrr']:.3f}")

    client.close()


if __name__ == "__main__":
    main()