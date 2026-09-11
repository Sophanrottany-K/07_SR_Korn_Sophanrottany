## Reflection

What worked well in my RAG application was retrieval and generation pipeline. The application successfully loads the existing documents, splits them into chunks, creates embeddings using `nomic-embed-text`, and stores them in ChromaDB. When a user asks a question, the system retrieves relevant chunks. This worked well for questions whose answers were contained in the documents. if user ask something that doesn't contain in the documnt our system can also generate answer saying "I don't have enough information in the documents to answer that question."

One thing that was harder than expected was setting up and connecting all the components correctly. In particular, configuring Python dependencies, Ollama models, embeddings, ChromaDB, and the different RAG modules required troubleshooting several issues before the complete pipeline worked.

For future improvement, I would add **re-ranking** to the retrieval process. Currently, ChromaDB retrieves the top relevant chunks based on embedding similarity. A re-ranking model could take these retrieved chunks and the user's question and score them again to determine which chunks are truly the most relevant. This could improve the quality of the context given to the generation model and, as a result, make the final answers more accurate and relevant.
