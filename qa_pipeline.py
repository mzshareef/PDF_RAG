from dotenv import load_dotenv
import os
import openai
import numpy as np
from embed_store import model

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def search_index(index, chunks, query, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def generate_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question based only on the following context:

Context:
{context}

Question: {query}
Answer:"""
    return prompt
    
def ask_openai(prompt):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content
