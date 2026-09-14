import json
import urllib.request

from retriever import SemanticRetriever


class RAGPipeline:
    def __init__(
        self,
        documents_directory: str,
        model_name: str = "llama3.2:3b",
    ):
        self.retriever = SemanticRetriever(documents_directory)
        self.model_name = model_name

    def generate_answer(self, query: str, top_k: int = 3) -> str:
        results = self.retriever.search(query, top_k=top_k)

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

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        request = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        return response_data["response"]


if __name__ == "__main__":
    rag = RAGPipeline("data/documents")

    # query = "How many annual leave days can I carry forward?"
    query = "Does AcmeTech provide employees with a gym membership?"

    answer = rag.generate_answer(query)

    print(f"\nQuestion: {query}\n")
    print("Answer:")
    print(answer)