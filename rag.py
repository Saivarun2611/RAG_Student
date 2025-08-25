import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No GEMINI_API_KEY found in .env file")

genai.configure(api_key=api_key)

# Load embeddings and index
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/course_index.faiss")

with open("data/processed_courses2.json", "r") as f:
    courses = json.load(f)

# Retrieval function
def search_courses(query, top_k=5):
    # encode query into embedding
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector).astype("float32")

    # search FAISS index
    scores, indices = index.search(query_vector, top_k)

    # collect course info 
    results = []
    for i, idx in enumerate(indices[0]):
        course = courses[idx]
        results.append({
            "rank": i+1,
            "course_number": course["course_number"],
            "title": course["title"],
            "description": course["description"],
            "score": float(scores[0][i])  # higher = better
        })
    return results

# RAG function
def rag_query(user_query, top_k=5):
    context = search_courses(user_query, top_k)

    prompt = f"""
You are an assistant that helps students explore Northeastern University's graduate Datascience courses.
Only use the provided context. Do not make up information.

User question:
{user_query}

Relevant course context:
{context}

Now provide a clear and helpful answer based only on the context above.
Help the students by explaining briefly how each course would be useful and relevant to their query.
If the answer is not in the context, say: "I couldn’t find this in the course catalog."
    """

    # using Gemini 2.0 Flash model
    llm = genai.GenerativeModel("gemini-2.0-flash",generation_config={'temperature':0.2})
    response = llm.generate_content(prompt)

    return response.text

# Check
if __name__ == "__main__":
    query = "I want to learn about machine learning and AI"
    answer = rag_query(query, top_k=5)
    print("Final Answer:\n")
    print(answer)
