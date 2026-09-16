from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class RetrievalRequest:
    query: str
    source: str
    top_k: int = 3
    filters: dict[str, Any] = field(default_factory=dict)
    retrieval_mode: RetrievalMode = RetrievalMode.SEMANTIC


@dataclass
class RetrievalResult:
    content: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    query: str
    source: str
    results: list[RetrievalResult]
    result_count: int