from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Raw retrieval scores are intentionally ignored because different
    retrieval systems may use incompatible scoring scales.
    """

    fused_scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (document, _) in enumerate(
            ranked_list,
            start=1,
        ):
            document_key = _document_key(document)

            documents[document_key] = document

            fused_scores[document_key] = (
                fused_scores.get(document_key, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    ranked_documents = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (
            documents[document_key],
            score,
        )
        for document_key, score in ranked_documents[:top_k]
    ]


def _document_key(
    document: Document,
) -> str:
    """
    Create a stable identity for a document chunk.
    """

    source = document.metadata.get(
        "source",
        ""
    )

    chunk_id = document.metadata.get(
        "chunk_id",
        ""
    )

    return f"{source}:{chunk_id}"