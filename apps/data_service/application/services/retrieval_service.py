from apps.data_service.domain.interfaces.retriever import RetrievalStrategy
from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
)


class RetrievalService:
    """Application service responsible for routing retrieval requests."""

    def __init__(self) -> None:
        self._strategies: dict[str, RetrievalStrategy] = {}

    def register(
        self,
        source: str,
        strategy: RetrievalStrategy,
    ) -> None:
        self._strategies[source] = strategy

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        strategy = self._strategies.get(request.source)

        if strategy is None:
            available_sources = ", ".join(
                sorted(self._strategies.keys())
            )

            raise ValueError(
                f"Unknown retrieval source: '{request.source}'. "
                f"Available sources: {available_sources}"
            )

        return strategy.retrieve(request)