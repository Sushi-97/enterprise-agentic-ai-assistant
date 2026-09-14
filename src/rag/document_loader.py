from pathlib import Path


def load_documents(directory: str) -> list[dict]:
    """
    Load Markdown documents from a directory.

    Returns:
        List of dictionaries containing document content and metadata.
    """
    document_path = Path(directory)

    documents = []

    for file_path in document_path.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": file_path.name,
                },
            }
        )

    return documents



if __name__ == "__main__":
    from section_chunker import chunk_by_sections

    docs = load_documents("data/documents")
    chunks = chunk_by_sections(docs)

    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} section chunks\n")

    for chunk in chunks[:8]:
        print(chunk["metadata"])
        print(chunk["content"][:300])
        print("-" * 50)