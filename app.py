import streamlit as st
import faiss
import pickle
import numpy as np
from embed_store import model
from qa_pipeline import search_index, generate_prompt, ask_openai

# Load prebuilt index and chunks
INDEX_PATH = "saved_index/index.faiss"
CHUNKS_PATH = "saved_index/chunks.pkl"

@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

index, chunks = load_index_and_chunks()

# UI
st.title("📚 Planning the Journey 2025 Document Q&A Bot")

question = st.text_input("Ask a question about the Hajj & Umrah 2025 document:")

if st.button("Get Answer") and question:
    relevant_chunks = search_index(index, chunks, question)
    prompt = generate_prompt(relevant_chunks, question)
    answer = ask_openai(prompt)
    st.write("### Answer:")
    st.write(answer)
