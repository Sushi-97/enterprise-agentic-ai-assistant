from fastapi import FastAPI
from qdrant_client import QdrantClient

from apps.data_service.api.routers import health, retrieval
from apps.data_service.api.routers.retrieval import configure_retrieval_service
from apps.data_service.application.services.retrieval_service import RetrievalService
from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.embeddings.local_sparse_embedding import (
    LocalSparseEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.qdrant_indexer import (
    QdrantDocumentIndexer,
)
from apps.data_service.infrastructure.retrieval.backends.qdrant_search_backend import (
    QdrantSearchBackend,
)
from apps.data_service.infrastructure.retrieval.local_document_retriever import (
    LocalDocumentRetrieval,
)


app = FastAPI(
    title="Enterprise Agentic AI - Data API",
    version="0.1.0",
)

qdrant_client = QdrantClient(path="data/qdrant")

dense_embedding_model = LocalEmbeddingModel()
sparse_embedding_model = LocalSparseEmbeddingModel()

QdrantDocumentIndexer(
    client=qdrant_client,
    collection_name="acmetech_policies",
    documents_directory="data/documents",
    embedding_model=dense_embedding_model,
    sparse_embedding_model=sparse_embedding_model,
).build()

search_backend = QdrantSearchBackend(
    client=qdrant_client,
    collection_name="acmetech_policies",
    embedding_model=dense_embedding_model,
    sparse_embedding_model=sparse_embedding_model,
)

retrieval_service = RetrievalService()

retrieval_service.register(
    "local_documents",
    LocalDocumentRetrieval(
        search_backend=search_backend,
    ),
)

configure_retrieval_service(retrieval_service)

app.include_router(health.router)
app.include_router(retrieval.router)