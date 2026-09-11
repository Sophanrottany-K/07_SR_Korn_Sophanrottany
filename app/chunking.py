from typing import List
from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Fixed-size chunking with overlap.
    """
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap  # step back so consecutive chunks share context

    return chunks