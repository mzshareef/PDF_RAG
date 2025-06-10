# PDF Q&A Bot

Simple Q&A chatbot that allows users to ask questions based only on the provided PDF. **. The app uses **RAG (Retrieval-Augmented Generation)** to extract relevant context from the PDF and then generate answers using **OpenAI's GPT-3.5 Turbo** model.

---

## Features

- Loads and chunks a single PDF document
- Converts text into embeddings using sentence-transformers
- Stores embeddings in a FAISS vector for similarity search
- On user query:
  - Grab the most relevant chunks
  - Send them to OpenAI to generate an answer based only on provided chunks

