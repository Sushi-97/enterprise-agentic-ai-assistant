from abc import ABC, abstractmethod

from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
)


class DataAPIClient(ABC):
    """Contract used by agents to access enterprise data capabilities."""

    @abstractmethod
    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        raise NotImplementedError
