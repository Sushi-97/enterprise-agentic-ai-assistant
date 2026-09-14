from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Dict]:
    """
    Split documents into overlapping text chunks.

    Args:
        documents: Output from document_loader.py
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []

    for document in documents:
        text = document["content"]
        source = document["metadata"]["source"]

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_id": chunk_id,
                    },
                }
            )

            chunk_id += 1
            start += chunk_size - chunk_overlap

    return chunks