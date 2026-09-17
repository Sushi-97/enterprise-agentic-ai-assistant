from apps.agent_service.application.services.knowledge_agent_service import (
    KnowledgeAgentService,
)
from apps.agent_service.infrastructure.data.http_data_api_client import (
    HTTPDataAPIClient,
)
from apps.agent_service.infrastructure.llm.ollama_llm_client import (
    OllamaLLMClient,
)
from evals.datasets.loader import load_dataset
from evals.evaluators.deterministic import evaluate_agent_result
from shared.llm.ollama_client import OllamaClient


DATASET_PATH = "evals/datasets/agents/knowledge_agent.yaml"


def main() -> None:
    dataset = load_dataset(DATASET_PATH)

    agent = KnowledgeAgentService(
        data_api_client=HTTPDataAPIClient(
            base_url="http://127.0.0.1:8001",
        ),
        llm_client=OllamaLLMClient(
            OllamaClient(model_name="llama3.2:3b")
        ),
    )

    passed = 0
    total = len(dataset["cases"])

    print("=" * 70)
    print(f"AGENT EVALUATION: {dataset['name']}")
    print("=" * 70)

    for case in dataset["cases"]:
        question = case["input"]["question"]

        actual = agent.answer(question)

        evaluation = evaluate_agent_result(
            actual=actual,
            expected=case["expected"],
            fallback_response=KnowledgeAgentService.FALLBACK_RESPONSE,
        )

        if evaluation.passed:
            passed += 1

        print("\n" + "-" * 70)
        print(
            f"{case['id']}: "
            f"{'PASS' if evaluation.passed else 'FAIL'}"
        )
        print("Question:", question)
        print("Answer:", actual.answer)
        print("Sources:", actual.sources)
        print("Grounded:", actual.grounded)
        print("Checks:", evaluation.checks)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Passed: {passed}/{total}")
    print(f"Pass Rate: {passed / total:.2%}")


if __name__ == "__main__":
    main()
