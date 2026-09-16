from apps.data_service.domain.interfaces.retriever import RetrievalStrategy
from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalResponse,
)
from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.vectorstores.local_vector_index import (
    LocalVectorIndex,
)


class LocalDocumentRetrieval(RetrievalStrategy):
    """
    Query-time retrieval over a pre-built local vector index.

    Responsibilities:
    - embed the incoming query
    - search the vector index
    - map infrastructure results to domain retrieval results
    """

    def __init__(
        self,
        index: LocalVectorIndex,
        embedding_model: LocalEmbeddingModel,
    ) -> None:
        self.index = index
        self.embedding_model = embedding_model

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        query_embedding = self.embedding_model.embed_query(
            request.query
        )

        raw_results = self.index.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
        )

        results = [
            RetrievalResult(
                content=document.page_content,
                score=score,
                source=document.metadata["source"],
                metadata=document.metadata,
            )
            for document, score in raw_results
        ]

        return RetrievalResponse(
            query=request.query,
            source=request.source,
            results=results,
            result_count=len(results),
        )