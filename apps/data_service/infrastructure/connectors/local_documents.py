from pathlib import Path

from langchain_core.documents import Document


def load_documents(directory: str) -> list[Document]:
    """Load Markdown files from a local directory as LangChain Documents."""

    documents: list[Document] = []

    for file_path in sorted(Path(directory).glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "source_type": "local_document",
                    "file_path": str(file_path),
                },
            )
        )

    return documents