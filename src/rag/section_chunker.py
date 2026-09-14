import re
from typing import List, Dict


def chunk_by_sections(documents: List[Dict]) -> List[Dict]:
    """
    Split Markdown documents by H2 headings.

    Keeps the document title as metadata instead of creating
    a separate title-only chunk.
    """
    chunks = []

    for document in documents:
        text = document["content"]
        source = document["metadata"]["source"]

        lines = text.splitlines()

        document_title = None

        if lines and lines[0].startswith("# "):
            document_title = lines[0].replace("# ", "").strip()
            text = "\n".join(lines[1:]).strip()

        sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)

        chunk_id = 0

        for section in sections:
            section = section.strip()

            if not section:
                continue

            chunks.append(
                {
                    "content": section,
                    "metadata": {
                        "source": source,
                        "document_title": document_title,
                        "chunk_id": chunk_id,
                    },
                }
            )

            chunk_id += 1

    return chunks