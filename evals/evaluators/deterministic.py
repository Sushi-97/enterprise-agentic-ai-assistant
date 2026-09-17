from dataclasses import dataclass
from typing import Any


@dataclass
class AgentEvaluationResult:
    passed: bool
    checks: dict[str, bool]


def evaluate_agent_result(
    actual: Any,
    expected: dict,
    fallback_response: str,
) -> AgentEvaluationResult:
    checks: dict[str, bool] = {}

    if "answer_contains" in expected:
        answer_lower = actual.answer.lower()
        checks["answer"] = all(
            value.lower() in answer_lower
            for value in expected["answer_contains"]
        )

    if "sources" in expected:
        checks["sources"] = all(
            source in actual.sources
            for source in expected["sources"]
        )

    if "grounded" in expected:
        checks["grounded"] = (
            actual.grounded == expected["grounded"]
        )

    if expected.get("abstain"):
        checks["abstention"] = (
            actual.answer == fallback_response
        )

    checks["no_error"] = actual.error is None

    return AgentEvaluationResult(
        passed=all(checks.values()),
        checks=checks,
    )


def find_relevant_rank(
    results: list[Any],
    expected_source: str,
    expected_section: str,
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if (
            result.source == expected_source
            and expected_section.lower() in result.content.lower()
        ):
            return rank

    return None


def calculate_ranking_metrics(
    ranks: list[int | None],
    top_k: int,
) -> dict[str, float]:
    total = len(ranks)

    if total == 0:
        return {
            "hit_at_1": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
        }

    return {
        "hit_at_1": sum(rank == 1 for rank in ranks) / total,
        "hit_at_k": sum(
            rank is not None and rank <= top_k
            for rank in ranks
        ) / total,
        "mrr": sum(
            1 / rank
            for rank in ranks
            if rank is not None
        ) / total,
    }
