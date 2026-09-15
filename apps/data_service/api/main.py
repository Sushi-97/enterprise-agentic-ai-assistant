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


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Agentic AI - Data API",
        version="0.1.0",
    )

    retrieval_service = RetrievalService()
    retrieval_service.register(
        "local_documents",
        LocalDocumentRetrieval("data/documents"),
    )

    configure_retrieval_service(retrieval_service)

    app.include_router(health_router)
    app.include_router(retrieval_router)

    return app


app = create_app()