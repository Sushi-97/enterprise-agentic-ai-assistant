from abc import ABC, abstractmethod

from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
)


class RetrievalStrategy(ABC):

    @abstractmethod
    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """Retrieve relevant information from a configured data source."""
        raise NotImplementedError