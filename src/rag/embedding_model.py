from sentence_transformers import SentenceTransformer


class LocalEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str):
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


if __name__ == "__main__":
    embedding_model = LocalEmbeddingModel()

    test_texts = [
        "Employees receive 20 days of annual leave.",
        "Business class may be approved for long international flights.",
    ]

    embeddings = embedding_model.embed_texts(test_texts)

    print("Embedding shape:", embeddings.shape)
    print("Vector dimension:", embeddings.shape[1])
    print("First 5 values:", embeddings[0][:5])