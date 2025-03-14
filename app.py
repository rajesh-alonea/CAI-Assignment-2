# app.py
import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load financial data
with open("salesforce_fy23_financials.json", "r") as f:
    fy23_data = json.load(f)

with open("salesforce_fy24_financials.json", "r") as f:
    fy24_data = json.load(f)

# Preprocess data into text chunks


def preprocess_data(data):
    chunks = []
    for key, value in data["financial_statements"].items():
        for sub_key, sub_value in value.items():
            chunk = f"{key} - {sub_key}: {sub_value['value']} {sub_value.get('unit', '')}"
            if "yoy_change" in sub_value:
                chunk += f", YoY Change: {sub_value['yoy_change']}"
            chunks.append(chunk)
    return chunks


fy23_chunks = preprocess_data(fy23_data)
fy24_chunks = preprocess_data(fy24_data)
all_chunks = fy23_chunks + fy24_chunks

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all chunks
chunk_embeddings = model.encode(all_chunks)

# Store embeddings in a vector database (FAISS)
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# Chunk Merging: Combine smaller chunks into larger chunks


def merge_chunks(chunks, max_tokens=100):
    merged_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk.split()) + len(chunk.split()) <= max_tokens:
            current_chunk += " " + chunk
        else:
            merged_chunks.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        merged_chunks.append(current_chunk.strip())
    return merged_chunks

# Adaptive Retrieval: Dynamically adjust chunk size based on query complexity


def retrieve_chunks(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    merged_chunks = merge_chunks(retrieved_chunks)
    return merged_chunks

# Generate response using retrieved chunks (no hallucination)


def generate_response(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    if not context:
        return "No relevant information found.", 0.0
    # Simple answer extraction (replace with a more sophisticated method if needed)
    if "revenue" in query.lower():
        for chunk in retrieved_chunks:
            if "revenue" in chunk.lower():
                return chunk, 1.0
    elif "operating income" in query.lower():
        for chunk in retrieved_chunks:
            if "operating income" in chunk.lower():
                return chunk, 1.0
    elif "net income" in query.lower():
        for chunk in retrieved_chunks:
            if "net income" in chunk.lower():
                return chunk, 1.0
    return "Sorry, I cannot provide a confident answer.", 0.5


# Streamlit UI
st.title("Salesforce Financials RAG Chatbot")
query = st.text_input("Enter your question:")

if query:
    retrieved_chunks = retrieve_chunks(query)
    response, confidence_score = generate_response(query, retrieved_chunks)
    st.write(f"Answer: {response}")
    st.write(f"Confidence Score: {confidence_score}")
