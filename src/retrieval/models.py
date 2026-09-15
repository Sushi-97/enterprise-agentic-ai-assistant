from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalRequest:
    query: str
    source: str
    top_k: int = 3
    filters: dict[str, Any] = field(default_factory=dict)


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