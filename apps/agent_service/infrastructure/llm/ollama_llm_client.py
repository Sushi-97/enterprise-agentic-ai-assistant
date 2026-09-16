from apps.agent_service.domain.interfaces.llm_client import (
    LLMClient,
    LLMResponse,
)
from shared.llm.ollama_client import OllamaClient


class OllamaLLMClient(LLMClient):
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def generate(self, prompt: str) -> LLMResponse:
        content, metrics = self.client.generate(prompt)

        metadata = {}

        if metrics is not None:
            metadata = (
                metrics.__dict__
                if hasattr(metrics, "__dict__")
                else {"metrics": metrics}
            )

        return LLMResponse(
            content=content,
            metadata=metadata,
        )
