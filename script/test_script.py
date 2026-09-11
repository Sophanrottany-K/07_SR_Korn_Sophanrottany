from app.retriever import retrieve
from app.generator import generate_answer


# ============================================================
# Test Questions
# ============================================================

questions = [
    # Questions whose answers should be inside your documents
    "how to set up email on mobilel device?",
    "how to reset PIN?",
    "how to Configure the VPN Connection?",

    #question which answer not in the documents
    "What is microservice?",

    # Question that should NOT be answered from the documents
    "Who is the president of the United States?",
]


# ============================================================
# Run Tests
# ============================================================

def run_tests():

    print("=" * 70)
    print("RAG APPLICATION TEST")
    print("=" * 70)

    for number, question in enumerate(questions, start=1):

        print("\n")
        print("=" * 70)
        print(f"QUESTION {number}")
        print("=" * 70)

        print(f"\nQuestion:")
        print(question)

        # ----------------------------------------------------
        # Retrieve relevant chunks
        # ----------------------------------------------------

        chunks = retrieve(question)

        print("\nRetrieved Chunks:")
        print("-" * 70)

        if not chunks:
            print("No chunks retrieved.")

        else:
            for i, chunk in enumerate(chunks, start=1):

                print(f"\nChunk {i}")
                print(f"Source: {chunk['source']}")
                print(f"Chunk Index: {chunk['chunk_index']}")
                print(f"Distance: {chunk['distance']:.4f}")
                print("\nText:")
                print(chunk["text"])

        # ----------------------------------------------------
        # Generate final answer
        # ----------------------------------------------------

        answer = generate_answer(
            question,
            chunks
        )

        print("\nFinal Answer:")
        print("-" * 70)
        print(answer)


if __name__ == "__main__":
    run_tests()