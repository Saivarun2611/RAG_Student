# query.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embeddings model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("data/course_index.faiss")

# Load metadata
with open("data/processed_courses2.json", "r") as f:
    courses = json.load(f)

def search_courses(query, top_k=5):
    # Convert query text to embedding
    query_vector = model.encode([query], normalize_embeddings=True)  # normalize
    query_vector = np.array(query_vector).astype("float32")

    # Search in FAISS index
    scores, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        course = courses[idx]
        results.append({
            "rank": i+1,
            "course_number": course["course_number"],
            "title": course["title"],
            "description": course["description"],
            "score": float(scores[0][i]) 
        })
    return results

if __name__ == "__main__":
    query = "I want to learn about machine learning and AI"
    results = search_courses(query, top_k=5)

    print(f"\nTop {len(results)} courses for query: '{query}'\n")
    for r in results:
        print(f"Rank {r['rank']} | {r['course_number']} - {r['title']}")
        print(f"Score (cosine similarity): {r['score']:.4f}")
        print(f"Description: {r['description'][:200]}...")
        print("-" * 80)
