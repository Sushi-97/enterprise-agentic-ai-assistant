from abc import ABC, abstractmethod

from retrieval.models import RetrievalRequest, RetrievalResponse


class RetrievalStrategy(ABC):

    @abstractmethod
    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        pass