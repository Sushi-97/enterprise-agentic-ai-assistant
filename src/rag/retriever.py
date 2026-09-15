import numpy as np

from document_loader import load_documents
from section_chunker import chunk_by_sections
from embedding_model import LocalEmbeddingModel

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from shared.cache.embedding_cache import EmbeddingCache


class SemanticRetriever:
    def __init__(self, documents_directory: str):
        self.embedding_model = LocalEmbeddingModel()

        documents = load_documents(documents_directory)
        self.chunks = chunk_by_sections(documents)

        chunk_texts = [chunk["content"] for chunk in self.chunks]

        self.cache = EmbeddingCache()

        fingerprint = self.cache.create_fingerprint(self.chunks)

        cached_embeddings = self.cache.load(fingerprint)

        if cached_embeddings is not None:
            print("Loading embeddings from cache...")
            self.chunk_embeddings = cached_embeddings
        else:
            print("Embedding cache miss. Generating embeddings...")

            self.chunk_embeddings = self.embedding_model.embed_texts(
                chunk_texts
            )

            self.cache.save(
                embeddings=self.chunk_embeddings,
                fingerprint=fingerprint,
            )

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.embed_query(query)

        similarity_scores = np.dot(
            self.chunk_embeddings,
            query_embedding,
        )

        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append(
                {
                    "score": float(similarity_scores[index]),
                    "content": self.chunks[index]["content"],
                    "metadata": self.chunks[index]["metadata"],
                }
            )

        return results


if __name__ == "__main__":
    retriever = SemanticRetriever("data/documents")

    query = "How many annual leave days can I carry forward?"

    results = retriever.search(query, top_k=3)

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(f"Result {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Title: {result['metadata']['document_title']}")
        print(result["content"])
        print("-" * 60)