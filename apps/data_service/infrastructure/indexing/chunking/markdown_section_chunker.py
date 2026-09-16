
import re

from langchain_core.documents import Document


def chunk_by_sections(documents: list[Document]) -> list[Document]:
    """
    Split Markdown documents by H2 headings.

    The H1 document title is stored as metadata instead of becoming
    a standalone chunk. Parent document metadata is propagated to
    every generated chunk.
    """

    chunks: list[Document] = []

    for document in documents:
        text = document.page_content
        lines = text.splitlines()

        document_title = None

        if lines and lines[0].startswith("# "):
            document_title = lines[0].removeprefix("# ").strip()
            text = "\n".join(lines[1:]).strip()

        sections = re.split(
            r"(?=^##\s)",
            text,
            flags=re.MULTILINE,
        )

        chunk_id = 0

        for section in sections:
            section = section.strip()

            if not section:
                continue

            metadata = {
                **document.metadata,
                "document_title": document_title,
                "chunk_id": chunk_id,
            }

            chunks.append(
                Document(
                    page_content=section,
                    metadata=metadata,
                )
            )

            chunk_id += 1

    return chunks