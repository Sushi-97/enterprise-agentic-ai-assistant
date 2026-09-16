from abc import ABC, abstractmethod

from shared.contracts.retrieval import (
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