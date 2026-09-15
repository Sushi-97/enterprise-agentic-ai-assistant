from fastapi import APIRouter, HTTPException

from apps.data_service.api.schemas.retrieval import (
    RetrievalAPIRequest,
    RetrievalAPIResult,
    RetrievalAPIResponse,
)

from apps.data_service.application.services.retrieval_service import (
    RetrievalService,
)
from apps.data_service.domain.models.retrieval import RetrievalRequest


router = APIRouter(
    prefix="/api/v1",
    tags=["retrieval"],
)


# Temporary dependency.
# We will move service construction out of the API layer in the next step.
retrieval_service: RetrievalService | None = None


def configure_retrieval_service(service: RetrievalService) -> None:
    global retrieval_service
    retrieval_service = service


@router.post(
    "/retrieval",
    response_model=RetrievalAPIResponse,
)
def retrieve(request: RetrievalAPIRequest):
    if retrieval_service is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval service is not configured.",
        )

    try:
        retrieval_request = RetrievalRequest(
            query=request.query,
            source=request.source,
            top_k=request.top_k,
            filters=request.filters,
        )

        response = retrieval_service.retrieve(retrieval_request)

        return RetrievalAPIResponse(
            query=response.query,
            source=response.source,
            result_count=response.result_count,
            results=[
                RetrievalAPIResult(
                    content=result.content,
                    score=result.score,
                    source=result.source,
                    metadata=result.metadata,
                )
                for result in response.results
            ],
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )