from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LLMMetrics:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    estimated_cost_usd: float
    cache_hit: bool = False
    agent: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
) -> float:
    """
    Estimate LLM API cost.

    Local Ollama models use zero pricing by default.
    Cloud model pricing can be supplied later.
    """

    input_cost = (
        prompt_tokens / 1_000_000
    ) * input_cost_per_million

    output_cost = (
        completion_tokens / 1_000_000
    ) * output_cost_per_million

    return input_cost + output_cost