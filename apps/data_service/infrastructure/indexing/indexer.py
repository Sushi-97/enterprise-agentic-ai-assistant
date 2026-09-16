from apps.data_service.infrastructure.connectors.local_documents import (
    load_documents,
)
from apps.data_service.infrastructure.indexing.chunking.markdown_section_chunker import (
    chunk_by_sections,
)
from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.vectorstores.local_vector_index import (
    LocalVectorIndex,
)
from shared.cache.embedding_cache import EmbeddingCache


class LocalDocumentIndexer:
    """
    Builds a searchable local vector index from Markdown documents.

    Index-time responsibilities:
    - load documents
    - chunk documents
    - generate/load embeddings
    - construct vector index
    """

    def __init__(
        self,
        documents_directory: str,
        embedding_model: LocalEmbeddingModel,
        cache: EmbeddingCache,
    ) -> None:
        self.documents_directory = documents_directory
        self.embedding_model = embedding_model
        self.cache = cache

    def build(self) -> LocalVectorIndex:
        documents = load_documents(
            self.documents_directory
        )

        chunks = chunk_by_sections(documents)

        fingerprint = self.cache.create_fingerprint(
            documents=chunks,
            embedding_model=self.embedding_model.model_name,
        )

        embeddings = self.cache.load(fingerprint)

        if embeddings is not None:
            print("Loading embeddings from cache...")
        else:
            print(
                "Embedding cache miss. "
                "Generating embeddings..."
            )

            embeddings = self.embedding_model.embed_texts(
                [
                    chunk.page_content
                    for chunk in chunks
                ]
            )

            self.cache.save(
                embeddings=embeddings,
                fingerprint=fingerprint,
                embedding_model=self.embedding_model.model_name,
            )

        return LocalVectorIndex(
            documents=chunks,
            embeddings=embeddings,
        )