from fastapi import FastAPI, Query
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load the FAISS index at startup
INDEX_PATH = "wikipedia_202307.index"
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
print("FAISS index loaded successfully.")

# Load the SentenceTransformer model at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer model loaded.")

@app.get("/search")
def search(query: str, top_k: int = 5):
    """
    Perform a similarity search on the FAISS index.

    :param query: The query string.
    :param top_k: Number of top results to return.
    :return: Indices and distances of the top results.
    """
    print(f"Processing query: {query}")
    query_vector = model.encode([query]).astype('float32')  # Generate query embedding
    distances, indices = index.search(query_vector, top_k)  # Perform FAISS search
    results = [{"index": int(idx), "distance": float(dist)} for idx, dist in zip(indices[0], distances[0])]
    return {"query": query, "results": results}
