from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeEvidence:
    source: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeAgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    evidence: list[KnowledgeEvidence] = field(default_factory=list)
    grounded: bool = False
    error: str | None = None