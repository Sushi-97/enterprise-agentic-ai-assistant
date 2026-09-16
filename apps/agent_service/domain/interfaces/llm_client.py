from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str
    metadata: dict[str, Any]


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError
