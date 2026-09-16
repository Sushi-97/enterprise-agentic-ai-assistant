from apps.data_service.domain.interfaces.search_backend import SearchBackend
from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResult,
)
from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.keyword.bm25_index import BM25Index
from apps.data_service.infrastructure.indexing.vectorstores.local_vector_index import (
    LocalVectorIndex,
)
from apps.data_service.infrastructure.retrieval.fusion.reciprocal_rank_fusion import (
    reciprocal_rank_fusion,
)


class LocalSearchBackend(SearchBackend):
    """Local baseline search using NumPy vectors + BM25 + RRF."""

    def __init__(
        self,
        vector_index: LocalVectorIndex,
        keyword_index: BM25Index,
        embedding_model: LocalEmbeddingModel,
    ) -> None:
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.embedding_model = embedding_model

    def search_semantic(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedding_model.embed_query(request.query)

        raw_results = self.vector_index.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
        )

        return self._to_retrieval_results(raw_results)

    def search_keyword(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        raw_results = self.keyword_index.search(
            query=request.query,
            top_k=request.top_k,
        )

        return self._to_retrieval_results(raw_results)

    def search_hybrid(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        candidate_k = max(request.top_k * 3, 10)

        query_embedding = self.embedding_model.embed_query(request.query)

        semantic_results = self.vector_index.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        keyword_results = self.keyword_index.search(
            query=request.query,
            top_k=candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[semantic_results, keyword_results],
            top_k=request.top_k,
        )

        return self._to_retrieval_results(fused_results)

    @staticmethod
    def _to_retrieval_results(
        raw_results,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content=document.page_content,
                score=score,
                source=document.metadata["source"],
                metadata=document.metadata,
            )
            for document, score in raw_results
        ]