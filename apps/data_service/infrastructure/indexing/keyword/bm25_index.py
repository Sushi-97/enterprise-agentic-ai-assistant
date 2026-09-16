import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Index:
    """
    In-memory BM25 keyword index for local development.

    Documents are tokenized once when the index is created.
    Queries are tokenized using the same strategy.
    """

    def __init__(
        self,
        documents: list[Document],
    ) -> None:
        if not documents:
            raise ValueError(
                "BM25 index requires at least one document."
            )

        self.documents = documents

        self.tokenized_documents = [
            self._tokenize(document.page_content)
            for document in documents
        ]

        self.index = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[Document, float]]:
        tokenized_query = self._tokenize(query)

        scores = self.index.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            (
                self.documents[index],
                float(scores[index]),
            )
            for index in ranked_indices
        ]