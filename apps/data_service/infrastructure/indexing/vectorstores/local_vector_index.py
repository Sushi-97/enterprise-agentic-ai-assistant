import numpy as np

from langchain_core.documents import Document


class LocalVectorIndex:
    """
    In-memory vector index for local development.

    Assumes stored embeddings and query embeddings are normalized,
    allowing dot product to represent cosine similarity.
    """

    def __init__(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        self.documents = documents
        self.embeddings = embeddings

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[Document, float]]:
        similarity_scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        top_indices = np.argsort(
            similarity_scores
        )[::-1][:top_k]

        return [
            (
                self.documents[index],
                float(similarity_scores[index]),
            )
            for index in top_indices
        ]