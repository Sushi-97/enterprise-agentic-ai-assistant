from abc import ABC, abstractmethod

from shared.contracts.retrieval import (
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
