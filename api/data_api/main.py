from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any

from retrieval.service import RetrievalService
from retrieval.local_document_retriever import LocalDocumentRetrieval
from retrieval.models import RetrievalRequest


app = FastAPI(
    title="Enterprise Agentic AI - Data API",
    version="0.1.0",
)


# -------------------------
# API contracts
# -------------------------

class RetrievalAPIRequest(BaseModel):
    query: str = Field(min_length=1)
    source: str
    top_k: int = Field(default=3, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalAPIResult(BaseModel):
    content: str
    score: float
    source: str
    metadata: dict[str, Any]


class RetrievalAPIResponse(BaseModel):
    query: str
    source: str
    result_count: int
    results: list[RetrievalAPIResult]


# -------------------------
# Retrieval configuration
# -------------------------

retrieval_service = RetrievalService()

retrieval_service.register(
    "local_documents",
    LocalDocumentRetrieval("data/documents"),
)


# -------------------------
# API endpoints
# -------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "data-api",
    }


@app.post(
    "/api/v1/retrieval",
    response_model=RetrievalAPIResponse,
)
def retrieve(request: RetrievalAPIRequest):

    try:
        retrieval_request = RetrievalRequest(
            query=request.query,
            source=request.source,
            top_k=request.top_k,
            filters=request.filters,
        )

        response = retrieval_service.retrieve(
            retrieval_request
        )

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