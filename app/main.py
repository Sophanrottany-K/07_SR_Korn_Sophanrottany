from app.ingestion import build_index
from app.pipeline import ask


def main():
    # Build the vector index
    print("Building index...")
    count = build_index()

    print(f"Indexed {count} chunks.")

    # Question-answering loop
    while True:
        query = input(
            "\nAsk a question (type 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            continue

        # Run the complete RAG pipeline
        answer = ask(query)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()