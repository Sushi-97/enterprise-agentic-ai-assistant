from typing import Any

from pydantic import BaseModel, Field

from apps.data_service.domain.models.retrieval import RetrievalMode

class RetrievalAPIRequest(BaseModel):
    query: str = Field(min_length=1)
    source: str
    top_k: int = Field(default=3, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_mode: RetrievalMode = RetrievalMode.SEMANTIC


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