import numpy as np

from retrieval.connectors.local_documents import load_documents
from retrieval.chunking.markdown_section_chunker import chunk_by_sections
from retrieval.embeddings.local_embedding import LocalEmbeddingModel

from shared.cache.embedding_cache import EmbeddingCache

from apps.data_service.domain.interfaces.retriever import RetrievalStrategy
from apps.data_service.domain.models.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalResponse,
)


class SemanticRetriever:
    def __init__(self, documents_directory: str):
        self.embedding_model = LocalEmbeddingModel()

        documents = load_documents(documents_directory)
        self.chunks = chunk_by_sections(documents)

        chunk_texts = [
            chunk["content"]
            for chunk in self.chunks
        ]

        self.cache = EmbeddingCache()

        fingerprint = self.cache.create_fingerprint(
            self.chunks
        )

        cached_embeddings = self.cache.load(
            fingerprint
        )

        if cached_embeddings is not None:
            print("Loading embeddings from cache...")
            self.chunk_embeddings = cached_embeddings

        else:
            print(
                "Embedding cache miss. "
                "Generating embeddings..."
            )

            self.chunk_embeddings = (
                self.embedding_model.embed_texts(
                    chunk_texts
                )
            )

            self.cache.save(
                embeddings=self.chunk_embeddings,
                fingerprint=fingerprint,
            )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ):
        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        similarity_scores = np.dot(
            self.chunk_embeddings,
            query_embedding,
        )

        top_indices = np.argsort(
            similarity_scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append(
                {
                    "score": float(
                        similarity_scores[index]
                    ),
                    "content": self.chunks[index][
                        "content"
                    ],
                    "metadata": self.chunks[index][
                        "metadata"
                    ],
                }
            )

        return results


class LocalDocumentRetrieval(RetrievalStrategy):
    def __init__(
        self,
        documents_directory: str,
    ):
        self.retriever = SemanticRetriever(
            documents_directory
        )

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:

        raw_results = self.retriever.search(
            query=request.query,
            top_k=request.top_k,
        )

        results = [
            RetrievalResult(
                content=result["content"],
                score=float(result["score"]),
                source=result["metadata"]["source"],
                metadata=result["metadata"],
            )
            for result in raw_results
        ]

        return RetrievalResponse(
            query=request.query,
            source=request.source,
            results=results,
            result_count=len(results),
        )