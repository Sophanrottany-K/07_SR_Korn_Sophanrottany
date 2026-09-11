from app.retriever import retrieve
from app.generator import generate_answer


def ask(query: str) -> str:
    """
    Connects retrieval and generation into one RAG flow.
    Question in -> grounded answer out.
    """

    # Retrieve relevant chunks
    chunks = retrieve(query)

    # Generate answer using the retrieved chunks
    answer = generate_answer(query, chunks)

    return answer