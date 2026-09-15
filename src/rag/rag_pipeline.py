import time

from retrieval.service import SemanticRetriever
from shared.llm.ollama_client import OllamaClient
from shared.observability.tracing import RequestTrace


class RAGPipeline:
    def __init__(
        self,
        documents_directory: str,
        model_name: str = "llama3.2:3b",
    ):
        self.retriever = SemanticRetriever(documents_directory)
        self.llm = OllamaClient(model_name=model_name)

    def generate_answer(
        self,
        query: str,
        top_k: int = 3,
    ):

        trace = RequestTrace(query=query)

        retrieval_start = time.perf_counter()

        results = self.retriever.search(
            query,
            top_k=top_k,
        )

        retrieval_latency = time.perf_counter() - retrieval_start

        trace.add_event(
            component="retrieval",
            latency_seconds=retrieval_latency,
            top_k=top_k,
            results_returned=len(results),
        )

        context_parts = []

        for result in results:
            source = result["metadata"]["source"]
            title = result["metadata"]["document_title"]
            content = result["content"]

            context_parts.append(
                f"Source: {source}\n"
                f"Document: {title}\n"
                f"Content:\n{content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are an enterprise knowledge assistant for AcmeTech.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not supported by the context, say:
   "I could not find enough information in the available company documents."
3. Keep the answer concise and clear.
4. Mention the source document(s) used.
5. Do not invent policies, numbers, dates, or approvals.

Context:
{context}

User question:
{query}

Answer:
""".strip()

        answer, metrics = self.llm.generate(
            prompt=prompt,
            agent="knowledge_agent",
        )

        trace.add_event(
            component="knowledge_agent",
            model=metrics.model,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.total_tokens,
            latency_seconds=metrics.latency_seconds,
            estimated_cost_usd=metrics.estimated_cost_usd,
            cache_hit=metrics.cache_hit,
        )

        trace_data = trace.finish()                     

        return answer, metrics, trace_data


if __name__ == "__main__":
    rag = RAGPipeline("data/documents")

    query = "Does AcmeTech provide employees with a gym membership?"

    answer, metrics, trace = rag.generate_answer(query)

    print(f"\nQuestion: {query}\n")

    print("Answer:")
    print(answer)

    print("\nMetrics:")
    print(metrics.to_dict())

    print("\nTrace:")
    print(trace)