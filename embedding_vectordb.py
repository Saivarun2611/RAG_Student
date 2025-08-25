import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load preprocessed data
with open("data/processed_courses2.json", "r") as f:
    courses = json.load(f)

# Generate embeddings
print('Generating Embeddings')
texts = [c["document"] for c in courses]
embeddings = model.encode(texts, normalize_embeddings=True,show_progress_bar=True)  # normalize

# Convert to float32 numpy cause FAISS needs float32 
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index (Inner Product = cosine similarity if normalized because Iam gonna use cosine similarity for retrival)
dim = embeddings.shape[1] # 384 for MiniLM
index = faiss.IndexFlatIP(dim)  # Inner Product
index.add(embeddings)

# Save index
faiss.write_index(index, "data/course_index.faiss")
with open("data/course_metadata.json", "w") as f:
    json.dump(courses, f, indent=2)

print(f"Indexed {len(courses)} courses with dimension {dim}")
