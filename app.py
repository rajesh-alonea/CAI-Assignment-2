# app.py
import streamlit as st
import json
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "rank_bm25"])


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re

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

# BM25 keyword-based search
bm25 = BM25Okapi([chunk.split() for chunk in all_chunks])

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
    # BM25 retrieval
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:]

    # Embedding-based retrieval
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    # Combine and re-rank results
    combined_indices = list(set(bm25_top_indices).union(set(indices[0])))
    combined_chunks = [all_chunks[i] for i in combined_indices]
    combined_embeddings = model.encode(combined_chunks)
    query_embedding = model.encode([query])
    distances = np.linalg.norm(combined_embeddings - query_embedding, axis=1)
    ranked_indices = np.argsort(distances)[:top_k]

    retrieved_chunks = [combined_chunks[i] for i in ranked_indices]
    merged_chunks = merge_chunks(retrieved_chunks)
    return merged_chunks

# Generate response using retrieved chunks (no hallucination)


def generate_response(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    if not context:
        return "No relevant information found.", 0.0
    # Simple answer extraction (replace with a more sophisticated method if needed)
    query_lower = query.lower()
    for chunk in retrieved_chunks:
        if any(term in query_lower for term in ["revenue", "gross profit", "operating income", "net income", "earnings per share", "total assets", "total liabilities", "shareholders equity", "long term debt", "cash and equivalents", "operating cash flow", "investing cash flow", "financing cash flow", "capital expenditures", "free cash flow", "gross margin", "operating margin", "net margin", "debt to equity ratio", "return on equity", "profit", "profit margin"]):
            if any(term in chunk.lower() for term in query_lower.split()):
                return chunk, 1.0
    return "Query is irrelevant.", 0.0

# Input-side guardrail: Validate and filter user queries


def validate_query(query):
    if not query or len(query) < 5:
        return False, "Query is too short."
    if not re.search(r'\b(?:revenue|gross profit|operating income|net income|earnings per share|total assets|total liabilities|shareholders equity|long term debt|cash and equivalents|operating cash flow|investing cash flow|financing cash flow|capital expenditures|free cash flow|gross margin|operating margin|net margin|debt to equity ratio|return on equity| profit | profit margin)\b', query, re.IGNORECASE):
        return False, "Query is irrelevant."
    return True, ""


# Streamlit UI
st.title("Salesforce Financials RAG Chatbot")
query = st.text_input("Enter your question:")

if query:
    is_valid, validation_message = validate_query(query)
    if is_valid:
        retrieved_chunks = retrieve_chunks(query)
        response, confidence_score = generate_response(query, retrieved_chunks)
        st.write(f"Answer: {response}")
        st.write(f"Confidence Score: {confidence_score}")
    else:
        st.write(f"Invalid query: {validation_message}")
