from fastembed import SparseTextEmbedding


class LocalSparseEmbeddingModel:
    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
    ) -> None:
        self.model_name = model_name
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed_texts(self, texts: list[str]):
        return list(self.model.embed(texts))

    def embed_query(self, query: str):
        return next(self.model.query_embed(query))