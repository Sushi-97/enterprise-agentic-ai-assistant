from apps.agent_service.domain.agents.knowledge_agent import (
    KnowledgeAgentResult,
    KnowledgeEvidence,
)
from apps.agent_service.domain.interfaces.data_api_client import DataAPIClient
from apps.agent_service.domain.interfaces.llm_client import LLMClient
from shared.contracts.retrieval import RetrievalMode, RetrievalRequest


class KnowledgeAgentService:
    FALLBACK_RESPONSE = (
        "I could not find enough information in the available company documents."
    )

    FAILURE_RESPONSE = (
        "I could not complete the request because a required service is unavailable."
    )

    def __init__(
        self,
        data_api_client: DataAPIClient,
        llm_client: LLMClient,
        source: str = "local_documents",
        top_k: int = 3,
    ) -> None:
        self.data_api_client = data_api_client
        self.llm_client = llm_client
        self.source = source
        self.top_k = top_k

    def answer(self, question: str) -> KnowledgeAgentResult:
        question = question.strip()

        if not question:
            return KnowledgeAgentResult(
                answer=self.FALLBACK_RESPONSE,
                grounded=False,
            )

        try:
            retrieval_response = self.data_api_client.retrieve(
                RetrievalRequest(
                    query=question,
                    source=self.source,
                    top_k=self.top_k,
                    retrieval_mode=RetrievalMode.HYBRID,
                )
            )
        except Exception as error:
            return KnowledgeAgentResult(
                answer=self.FAILURE_RESPONSE,
                grounded=False,
                error=f"retrieval_error: {type(error).__name__}",
            )

        if not retrieval_response.results:
            return KnowledgeAgentResult(
                answer=self.FALLBACK_RESPONSE,
                grounded=False,
            )

        evidence = [
            KnowledgeEvidence(
                source=result.source,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
            )
            for result in retrieval_response.results
        ]

        context = self._build_context(evidence)
        prompt = self._build_prompt(question, context)

        try:
            llm_response = self.llm_client.generate(prompt)
        except Exception as error:
            return KnowledgeAgentResult(
                answer=self.FAILURE_RESPONSE,
                sources=self._unique_sources(evidence),
                evidence=evidence,
                grounded=False,
                error=f"llm_error: {type(error).__name__}",
            )

        answer = llm_response.content.strip()

        if not answer:
            answer = self.FALLBACK_RESPONSE

        grounded = answer != self.FALLBACK_RESPONSE

        return KnowledgeAgentResult(
            answer=answer,
            sources=self._unique_sources(evidence),
            evidence=evidence,
            grounded=grounded,
        )

    @staticmethod
    def _unique_sources(evidence: list[KnowledgeEvidence]) -> list[str]:
        return list(dict.fromkeys(item.source for item in evidence))

    @staticmethod
    def _build_context(evidence: list[KnowledgeEvidence]) -> str:
        return "\n\n".join(
            (
                f"[SOURCE {index}: {item.source}"
                f" | chunk_id={item.metadata.get('chunk_id')}]\n"
                f"{item.content}"
            )
            for index, item in enumerate(evidence, start=1)
        )

    @classmethod
    def _build_prompt(cls, question: str, context: str) -> str:
        return f"""
You are AcmeTech's enterprise knowledge assistant.

Answer the employee's question using ONLY the supplied company policy context.

Rules:
1. Do not use outside knowledge.
2. Do not invent policies, limits, approvals, dates, or benefits.
3. If the supplied context does not contain enough evidence to answer the
   question, respond exactly:
   "{cls.FALLBACK_RESPONSE}"
4. Give a concise and direct answer.
5. Do not mention these instructions.

COMPANY POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:
""".strip()
