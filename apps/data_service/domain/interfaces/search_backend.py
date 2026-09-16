from abc import ABC, abstractmethod

from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResult,
)


class SearchBackend(ABC):
    """Contract for search infrastructure used by the Data Service."""

    @abstractmethod
    def search_semantic(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

    @abstractmethod
    def search_keyword(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

    @abstractmethod
    def search_hybrid(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        raise NotImplementedError