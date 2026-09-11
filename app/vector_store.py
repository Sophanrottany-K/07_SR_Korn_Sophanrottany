import chromadb

from app.config import CHROMA_DB_DIR, COLLECTION_NAME


def get_collection():
    """Get or create the ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def add_chunks(
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
):
    """Save document chunks and their embeddings to ChromaDB."""

    collection = get_collection()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search(
    query_embedding: list[float],
    top_k: int = 4,
):
    """Search ChromaDB for the most similar chunks."""

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results