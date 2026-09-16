from apps.agent_service.application.services.knowledge_agent_service import (
    KnowledgeAgentService,
)
from apps.agent_service.infrastructure.data.http_data_api_client import (
    HTTPDataAPIClient,
)
from apps.agent_service.infrastructure.llm.ollama_llm_client import (
    OllamaLLMClient,
)
from shared.llm.ollama_client import OllamaClient


TEST_CASES = [
    {
        "question": "How many annual leave days can I carry forward?",
        "expected_text": "5",
        "expected_source": "leave_policy.md",
        "should_be_grounded": True,
    },
    {
        "question": "How many paid sick leave days do employees receive?",
        "expected_text": "10",
        "expected_source": "leave_policy.md",
        "should_be_grounded": True,
    },
    {
        "question": "What is the daily meal allowance during business travel?",
        "expected_text": "60",
        "expected_source": "travel_policy.md",
        "should_be_grounded": True,
    },
    {
        "question": "When do I need a receipt for an expense?",
        "expected_text": "25",
        "expected_source": "expense_policy.md",
        "should_be_grounded": True,
    },
    {
        "question": "Who needs to approve an expense report above $500?",
        "expected_text": "Finance",
        "expected_source": "expense_policy.md",
        "should_be_grounded": True,
    },
    {
        "question": "Does AcmeTech provide employees with a gym membership?",
        "expected_text": KnowledgeAgentService.FALLBACK_RESPONSE,
        "expected_source": None,
        "should_be_grounded": False,
    },
]


def main() -> None:
    agent = KnowledgeAgentService(
        data_api_client=HTTPDataAPIClient(
            base_url="http://127.0.0.1:8001",
        ),
        llm_client=OllamaLLMClient(
            OllamaClient(model_name="llama3.2:3b")
        ),
    )

    passed = 0

    for index, test_case in enumerate(TEST_CASES, start=1):
        result = agent.answer(test_case["question"])

        answer_match = (
            test_case["expected_text"].lower()
            in result.answer.lower()
        )

        source_match = (
            test_case["expected_source"] is None
            or test_case["expected_source"] in result.sources
        )

        grounding_match = (
            result.grounded
            == test_case["should_be_grounded"]
        )

        case_passed = (
            answer_match
            and source_match
            and grounding_match
            and result.error is None
        )

        if case_passed:
            passed += 1

        print("\n" + "=" * 70)
        print(f"CASE {index}: {'PASS' if case_passed else 'FAIL'}")
        print("Question:", test_case["question"])
        print("Answer:", result.answer)
        print("Sources:", result.sources)
        print("Grounded:", result.grounded)
        print("Error:", result.error)

    total = len(TEST_CASES)

    print("\n" + "=" * 70)
    print("KNOWLEDGE AGENT EVALUATION")
    print("=" * 70)
    print(f"Passed: {passed}/{total}")
    print(f"Pass Rate: {passed / total:.2%}")


if __name__ == "__main__":
    main()
