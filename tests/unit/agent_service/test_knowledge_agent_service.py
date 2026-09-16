from apps.agent_service.application.services.knowledge_agent_service import (
    KnowledgeAgentService,
)
from apps.agent_service.domain.interfaces.data_api_client import DataAPIClient
from apps.agent_service.domain.interfaces.llm_client import LLMClient, LLMResponse
from shared.contracts.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)


class FakeDataAPIClient(DataAPIClient):
    def __init__(
        self,
        response: RetrievalResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.last_request: RetrievalRequest | None = None

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        self.last_request = request

        if self.error:
            raise self.error

        if self.response is None:
            raise RuntimeError("Fake response was not configured.")

        return self.response


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        content: str = "",
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt

        if self.error:
            raise self.error

        return LLMResponse(
            content=self.content,
            metadata={},
        )


def retrieval_response(
    results: list[RetrievalResult],
) -> RetrievalResponse:
    return RetrievalResponse(
        query="test question",
        source="local_documents",
        results=results,
        result_count=len(results),
    )


def test_supported_question_returns_grounded_answer():
    data_client = FakeDataAPIClient(
        response=retrieval_response(
            [
                RetrievalResult(
                    content=(
                        "Employees may carry forward a maximum "
                        "of 5 unused annual leave days."
                    ),
                    score=1.0,
                    source="leave_policy.md",
                    metadata={"chunk_id": 2},
                )
            ]
        )
    )

    llm_client = FakeLLMClient(
        content="Employees may carry forward a maximum of 5 days."
    )

    agent = KnowledgeAgentService(data_client, llm_client)

    result = agent.answer(
        "How many annual leave days can I carry forward?"
    )

    assert result.grounded is True
    assert result.error is None
    assert result.sources == ["leave_policy.md"]
    assert len(result.evidence) == 1
    assert "5 days" in result.answer

    assert data_client.last_request is not None
    assert data_client.last_request.retrieval_mode.value == "hybrid"

    assert llm_client.last_prompt is not None
    assert "5 unused annual leave days" in llm_client.last_prompt


def test_unsupported_question_returns_fallback():
    data_client = FakeDataAPIClient(
        response=retrieval_response(
            [
                RetrievalResult(
                    content="Employees may claim approved business expenses.",
                    score=0.625,
                    source="expense_policy.md",
                    metadata={"chunk_id": 6},
                )
            ]
        )
    )

    llm_client = FakeLLMClient(
        content=KnowledgeAgentService.FALLBACK_RESPONSE
    )

    agent = KnowledgeAgentService(data_client, llm_client)

    result = agent.answer(
        "Does AcmeTech provide employees with a gym membership?"
    )

    assert result.answer == KnowledgeAgentService.FALLBACK_RESPONSE
    assert result.grounded is False
    assert result.error is None


def test_no_retrieval_results_skips_llm():
    data_client = FakeDataAPIClient(
        response=retrieval_response([])
    )
    llm_client = FakeLLMClient(
        content="This should never be returned."
    )

    agent = KnowledgeAgentService(data_client, llm_client)

    result = agent.answer("Unknown policy")

    assert result.answer == KnowledgeAgentService.FALLBACK_RESPONSE
    assert result.grounded is False
    assert llm_client.last_prompt is None


def test_data_api_failure_is_controlled():
    data_client = FakeDataAPIClient(
        error=RuntimeError("Data API unavailable")
    )
    llm_client = FakeLLMClient()

    agent = KnowledgeAgentService(data_client, llm_client)

    result = agent.answer("What is the leave policy?")

    assert result.answer == KnowledgeAgentService.FAILURE_RESPONSE
    assert result.grounded is False
    assert result.error == "retrieval_error: RuntimeError"


def test_llm_failure_is_controlled():
    data_client = FakeDataAPIClient(
        response=retrieval_response(
            [
                RetrievalResult(
                    content="Annual leave policy.",
                    score=1.0,
                    source="leave_policy.md",
                    metadata={"chunk_id": 1},
                )
            ]
        )
    )

    llm_client = FakeLLMClient(
        error=RuntimeError("LLM unavailable")
    )

    agent = KnowledgeAgentService(data_client, llm_client)

    result = agent.answer("What is the annual leave policy?")

    assert result.answer == KnowledgeAgentService.FAILURE_RESPONSE
    assert result.grounded is False
    assert result.error == "llm_error: RuntimeError"
    assert result.sources == ["leave_policy.md"]
    assert len(result.evidence) == 1


def test_empty_question_returns_fallback_without_dependencies():
    data_client = FakeDataAPIClient(
        error=RuntimeError("Should never be called")
    )
    llm_client = FakeLLMClient(
        error=RuntimeError("Should never be called")
    )

    agent = KnowledgeAgentService(data_client, llm_client)

    result = agent.answer("   ")

    assert result.answer == KnowledgeAgentService.FALLBACK_RESPONSE
    assert result.grounded is False
    assert result.error is None
    assert data_client.last_request is None
    assert llm_client.last_prompt is None
