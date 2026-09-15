import hashlib
import json
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, cache_directory: str = ".cache/embeddings"):
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        self.embeddings_file = self.cache_directory / "embeddings.npy"
        self.metadata_file = self.cache_directory / "metadata.json"

    def create_fingerprint(self, chunks: list[dict]) -> str:
        """
        Create a deterministic fingerprint of the knowledge-base chunks.

        If chunk content changes, the fingerprint changes and the
        existing embedding cache becomes invalid.
        """
        hasher = hashlib.sha256()

        for chunk in chunks:
            hasher.update(chunk["content"].encode("utf-8"))

            metadata = json.dumps(
                chunk["metadata"],
                sort_keys=True,
            )

            hasher.update(metadata.encode("utf-8"))

        return hasher.hexdigest()

    def load(self, fingerprint: str):
        """
        Load cached embeddings if the cache exists and its fingerprint
        matches the current knowledge base.
        """
        if not self.embeddings_file.exists():
            return None

        if not self.metadata_file.exists():
            return None

        with self.metadata_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata.get("fingerprint") != fingerprint:
            return None

        return np.load(self.embeddings_file)

    def save(self, embeddings, fingerprint: str):
        """
        Persist embeddings and the corresponding knowledge-base
        fingerprint.
        """
        np.save(self.embeddings_file, embeddings)

        metadata = {
            "fingerprint": fingerprint,
            "embedding_count": len(embeddings),
        }

        with self.metadata_file.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)