import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbeddingModel:
    """
    Local embedding provider backed by SentenceTransformers.

    Embeddings are normalized so dot-product similarity is equivalent
    to cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )