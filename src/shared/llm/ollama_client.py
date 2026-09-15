import json
import time
import urllib.request

from shared.observability.llm_metrics import LLMMetrics, calculate_cost

class OllamaClient:
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        agent: str | None = None,
    ) -> tuple[str, LLMMetrics]:

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_time = time.perf_counter()

        with urllib.request.urlopen(request) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

        latency = time.perf_counter() - start_time

        prompt_tokens = response_data.get(
            "prompt_eval_count", 0
        )

        completion_tokens = response_data.get(
            "eval_count", 0
        )

        total_tokens = prompt_tokens + completion_tokens

        cost = calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        metrics = LLMMetrics(
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            cache_hit=False,
            agent=agent,
        )

        return response_data["response"], metrics
