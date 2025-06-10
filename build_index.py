# build_index.py
from ingest import load_pdf, split_text
from embed_store import embed_texts, create_faiss_index
import faiss
import numpy as np
import pickle
import os

PDF_PATH = "data/PTJ-2025.pdf"
INDEX_PATH = "saved_index/index.faiss"
CHUNKS_PATH = "saved_index/chunks.pkl"

os.makedirs("saved_index", exist_ok=True)

# Step 1: Load and chunk PDF
text = load_pdf(PDF_PATH)
chunks = split_text(text)

# Step 2: Embed chunks
embeddings = embed_texts(chunks)

# Step 3: Create and save FAISS index
index = create_faiss_index(embeddings)

faiss.write_index(index, INDEX_PATH)

# Save chunks so we can load later
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("✅ Index and chunks saved successfully!")
