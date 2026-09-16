import hashlib
import json
from pathlib import Path

import numpy as np
from langchain_core.documents import Document


class EmbeddingCache:
    """
    Persistent cache for document embeddings.

    Cache validity depends on:
    - document content
    - document metadata
    - embedding model
    """

    def __init__(
        self,
        cache_directory: str = ".cache/embeddings",
    ) -> None:
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embeddings_file = (
            self.cache_directory / "embeddings.npy"
        )
        self.metadata_file = (
            self.cache_directory / "metadata.json"
        )

    def create_fingerprint(
        self,
        documents: list[Document],
        embedding_model: str,
    ) -> str:
        """
        Create a deterministic fingerprint for the indexed documents
        and embedding configuration.
        """

        hasher = hashlib.sha256()

        # Prevent embeddings from one model being reused by another.
        hasher.update(
            embedding_model.encode("utf-8")
        )

        for document in documents:
            hasher.update(
                document.page_content.encode("utf-8")
            )

            metadata = json.dumps(
                document.metadata,
                sort_keys=True,
                default=str,
            )

            hasher.update(
                metadata.encode("utf-8")
            )

        return hasher.hexdigest()

    def load(
        self,
        fingerprint: str,
    ) -> np.ndarray | None:
        """
        Load embeddings only when the stored fingerprint matches.
        """

        if not self.embeddings_file.exists():
            return None

        if not self.metadata_file.exists():
            return None

        with self.metadata_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if metadata.get("fingerprint") != fingerprint:
            return None

        return np.load(
            self.embeddings_file,
            allow_pickle=False,
        )

    def save(
        self,
        embeddings: np.ndarray,
        fingerprint: str,
        embedding_model: str,
    ) -> None:
        """Persist embeddings and cache metadata."""

        np.save(
            self.embeddings_file,
            embeddings,
        )

        metadata = {
            "fingerprint": fingerprint,
            "embedding_model": embedding_model,
            "embedding_count": len(embeddings),
            "embedding_dimension": (
                int(embeddings.shape[1])
                if embeddings.ndim == 2
                else None
            ),
        }

        with self.metadata_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )