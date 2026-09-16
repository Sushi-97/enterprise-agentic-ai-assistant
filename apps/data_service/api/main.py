from fastapi import FastAPI

from apps.data_service.api.routers.health import router as health_router
from apps.data_service.api.routers.retrieval import (
    configure_retrieval_service,
    router as retrieval_router,
)

from apps.data_service.infrastructure.retrieval.local_document_retriever import (
    LocalDocumentRetrieval,
)
from apps.data_service.application.services.retrieval_service import (
    RetrievalService,
)

from apps.data_service.infrastructure.indexing.embeddings.local_embedding import (
    LocalEmbeddingModel,
)
from apps.data_service.infrastructure.indexing.indexer import (
    LocalDocumentIndexer,
)
from shared.cache.embedding_cache import EmbeddingCache

def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Agentic AI - Data API",
        version="0.1.0",
    )

    embedding_model = LocalEmbeddingModel()

    indexer = LocalDocumentIndexer(
        documents_directory="data/documents",
        embedding_model=embedding_model,
        cache=EmbeddingCache(),
    )

    local_index = indexer.build()

    retrieval_service = RetrievalService()

    retrieval_service.register(
        "local_documents",
        LocalDocumentRetrieval(
            index=local_index,
            embedding_model=embedding_model,
        ),
    )

    configure_retrieval_service(retrieval_service)

    app.include_router(health_router)
    app.include_router(retrieval_router)

    return app


app = create_app()