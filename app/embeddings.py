from typing import List
import ollama
from app.config import EMBED_MODEL

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of strings. Returns one vector per input string, same order."""
    if not texts:
        return []
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return list(response["embeddings"])


def embed_query(text: str) -> List[float]:
    """Convenience wrapper for embedding a single piece of text (e.g. a question)."""
    return embed_texts([text])[0]