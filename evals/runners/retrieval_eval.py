from qdrant_client import QdrantClient

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
from evals.datasets.loader import load_dataset
from evals.evaluators.deterministic import (
    calculate_ranking_metrics,
    find_relevant_rank,
)
from shared.contracts.retrieval import (
    RetrievalMode,
    RetrievalRequest,
)


DATASET_PATH = "evals/datasets/retrieval/policy_retrieval.yaml"


def search(
    backend: QdrantSearchBackend,
    request: RetrievalRequest,
):
    if request.retrieval_mode == RetrievalMode.SEMANTIC:
        return backend.search_semantic(request)

    if request.retrieval_mode == RetrievalMode.KEYWORD:
        return backend.search_keyword(request)

    if request.retrieval_mode == RetrievalMode.HYBRID:
        return backend.search_hybrid(request)

    raise ValueError(
        f"Unsupported retrieval mode: {request.retrieval_mode}"
    )


def evaluate_mode(
    backend: QdrantSearchBackend,
    dataset: dict,
    mode: RetrievalMode,
) -> dict[str, float]:
    ranks: list[int | None] = []

    source = dataset["source"]
    top_k = dataset["top_k"]

    for case in dataset["cases"]:
        request = RetrievalRequest(
            query=case["question"],
            source=source,
            top_k=top_k,
            retrieval_mode=mode,
        )

        results = search(backend, request)

        rank = find_relevant_rank(
            results=results,
            expected_source=case["expected"]["source"],
            expected_section=case["expected"]["section"],
        )

        ranks.append(rank)

        print(
            f"{case['id']}: "
            f"{'rank ' + str(rank) if rank else 'NOT FOUND'}"
        )

    return calculate_ranking_metrics(
        ranks=ranks,
        top_k=top_k,
    )


def main() -> None:
    dataset = load_dataset(DATASET_PATH)

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
    print(f"DATASET: {dataset['name']}")
    print("=" * 65)

    for mode in RetrievalMode:
        print(f"\n{mode.value.upper()}")
        print("-" * 40)

        metrics = evaluate_mode(
            backend=backend,
            dataset=dataset,
            mode=mode,
        )

        print(f"\nHit Rate@1: {metrics['hit_at_1']:.2%}")
        print(
            f"Hit Rate@{dataset['top_k']}: "
            f"{metrics['hit_at_k']:.2%}"
        )
        print(f"MRR:        {metrics['mrr']:.3f}")

    client.close()


if __name__ == "__main__":
    main()
