from qdrant_client import QdrantClient, models

from apps.data_service.infrastructure.connectors.local_documents import (
    load_documents,
)
from apps.data_service.infrastructure.indexing.chunking.markdown_section_chunker import (
    chunk_by_sections,
)
from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.embeddings.local_sparse_embedding import (
    LocalSparseEmbeddingModel,
)


class QdrantDocumentIndexer:
    """
    Builds the Qdrant index for local enterprise documents.

    Each chunk is indexed with:
    - a dense semantic vector
    - a sparse lexical vector
    - filterable metadata
    """

    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "sparse"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        documents_directory: str,
        embedding_model: LocalEmbeddingModel,
        sparse_embedding_model: LocalSparseEmbeddingModel,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.documents_directory = documents_directory
        self.embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model

    def build(self) -> int:
        # 1. Load source documents
        documents = load_documents(self.documents_directory)

        # 2. Split documents into retrieval chunks
        chunks = chunk_by_sections(documents)

        if not chunks:
            raise ValueError(
                f"No document chunks found in: {self.documents_directory}"
            )

        texts = [chunk.page_content for chunk in chunks]

        # 3. Generate dense semantic embeddings
        dense_embeddings = self.embedding_model.embed_texts(texts)

        # 4. Generate sparse lexical embeddings
        sparse_embeddings = self.sparse_embedding_model.embed_texts(texts)

        # 5. Recreate the collection for the current development index
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.DENSE_VECTOR: models.VectorParams(
                    size=dense_embeddings.shape[1],
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        # 6. Create payload indexes for fields used in filtering
        for field_name in [
            "source",
            "source_type",
            "document_title",
        ]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        # 7. Convert chunks and embeddings into Qdrant points
        points = []

        for index, (chunk, dense, sparse) in enumerate(
            zip(
                chunks,
                dense_embeddings,
                sparse_embeddings,
            )
        ):
            points.append(
                models.PointStruct(
                    id=index,
                    vector={
                        self.DENSE_VECTOR: dense.tolist(),
                        self.SPARSE_VECTOR: models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={
                        "content": chunk.page_content,
                        "source": chunk.metadata["source"],
                        "source_type": chunk.metadata.get("source_type"),
                        "document_title": chunk.metadata.get(
                            "document_title"
                        ),
                        "chunk_id": chunk.metadata.get("chunk_id"),
                        "metadata": chunk.metadata,
                    },
                )
            )

        # 8. Persist points into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)