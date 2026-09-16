from qdrant_client import QdrantClient, models

from apps.data_service.domain.interfaces.search_backend import SearchBackend
from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResult,
)
from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.embeddings.local_sparse_embedding import (
    LocalSparseEmbeddingModel,
)


class QdrantSearchBackend(SearchBackend):
    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "sparse"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: LocalEmbeddingModel,
        sparse_embedding_model: LocalSparseEmbeddingModel,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model

    def search_semantic(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        query = self.embedding_model.embed_query(request.query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query.tolist(),
            using=self.DENSE_VECTOR,
            query_filter=self._build_filter(request.filters),
            limit=request.top_k,
            with_payload=True,
        )

        return self._to_retrieval_results(response.points)

    def search_keyword(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        sparse = self.sparse_embedding_model.embed_query(request.query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            ),
            using=self.SPARSE_VECTOR,
            query_filter=self._build_filter(request.filters),
            limit=request.top_k,
            with_payload=True,
        )

        return self._to_retrieval_results(response.points)

    def search_hybrid(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        dense = self.embedding_model.embed_query(request.query)
        sparse = self.sparse_embedding_model.embed_query(request.query)

        candidate_k = max(request.top_k * 3, 10)

        query_filter = self._build_filter(request.filters)

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense.tolist(),
                    using=self.DENSE_VECTOR,
                    filter=query_filter,
                    limit=candidate_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                    using=self.SPARSE_VECTOR,
                    filter=query_filter,
                    limit=candidate_k,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            limit=request.top_k,
            with_payload=True,
        )

        return self._to_retrieval_results(response.points)

    @staticmethod
    def _build_filter(
        filters: dict,
    ) -> models.Filter | None:
        if not filters:
            return None

        conditions: list[models.Condition] = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]

        return models.Filter(must=conditions)

    @staticmethod
    def _to_retrieval_results(points) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content=point.payload["content"],
                score=float(point.score),
                source=point.payload["source"],
                metadata=point.payload.get("metadata", {}),
            )
            for point in points
        ]