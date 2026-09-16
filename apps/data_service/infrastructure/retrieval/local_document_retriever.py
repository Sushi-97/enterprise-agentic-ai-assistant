from apps.data_service.domain.interfaces.retriever import RetrievalStrategy
from apps.data_service.domain.interfaces.search_backend import SearchBackend
from shared.contracts.retrieval import (
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
)


class LocalDocumentRetrieval(RetrievalStrategy):
    def __init__(self, search_backend: SearchBackend) -> None:
        self.search_backend = search_backend

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        if request.retrieval_mode == RetrievalMode.SEMANTIC:
            results = self.search_backend.search_semantic(request)

        elif request.retrieval_mode == RetrievalMode.KEYWORD:
            results = self.search_backend.search_keyword(request)

        elif request.retrieval_mode == RetrievalMode.HYBRID:
            results = self.search_backend.search_hybrid(request)

        else:
            raise ValueError(
                f"Unsupported retrieval mode: {request.retrieval_mode}"
            )

        return RetrievalResponse(
            query=request.query,
            source=request.source,
            results=results,
            result_count=len(results),
        )