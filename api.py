# api.py
import os
import json
import faiss
import numpy as np
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# Startup: load env & configure

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.0-flash"  

# Load retriever

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

INDEX_PATH = "data/course_index.faiss"
META_PATH = "data/processed_courses2.json"

if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError("Missing FAISS index or metadata. Run embeddingvectordb.py first.")

faiss_index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    COURSES = json.load(f)


# FastAPI app setup

app = FastAPI(title="Course RAG API", version="1.0.0", description="RAG for NEU DS courses")

# CORS so a local frontend  can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic request/response models

class RetrieveRequest(BaseModel):
    question: str = Field(..., description="Student's query")
    top_k: int = Field(5, ge=1, le=20, description="How many courses to retrieve")

class CourseOut(BaseModel):
    rank: int
    course_number: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    score: float

class RetrieveResponse(BaseModel):
    courses: List[CourseOut]

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=1.0)

class AskResponse(BaseModel):
    model: str
    answer: str
    courses: List[CourseOut]


# Core retrieval 
def search_courses(question: str, top_k: int = 5) -> List[CourseOut]:
    q_vec = embed_model.encode([question], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")
    scores, indices = faiss_index.search(q_vec, top_k)

    results: List[CourseOut] = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(COURSES):
            # FAISS sometimes returns -1 if it can't find neighbors
            # skip invalid indices
            continue
        c = COURSES[idx]
        results.append(
            CourseOut(
                rank=i + 1,
                course_number=c.get("course_number"),
                title=c.get("title"),
                description=c.get("description"),
                url=c.get("url"),
                score=float(scores[0][i]),
            )
        )
    return results


# Prompt builder 
def build_prompt(user_query: str, retrieved: List[CourseOut]) -> str:
    context_lines = []
    for c in retrieved:
        # keep consistent, no chunking
        context_lines.append(
            f"{c.course_number} - {c.title}\nDescription: {c.description}\nURL: {c.url}"
        )
    context = "\n\n".join(context_lines)

    prompt = f"""
You are an assistant that helps students explore Northeastern University's graduate Data Science courses.
Only use the provided course context. Do not make up information.

User question:
{user_query}

Relevant course context:
{context}

Instructions:
- Base your answer ONLY on the context above.
- Briefly explain why each suggested course is relevant.
- If the information is not present, reply: "I couldn’t find this in the course catalog."
"""
    return prompt.strip()


# Routes

@app.get("/")
def home():
    return {"message": "Welcome to the RAG API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    try:
        courses = search_courses(req.question, req.top_k)
        return {"courses": courses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        courses = search_courses(req.question, req.top_k)
        prompt = build_prompt(req.question, courses)

        # Configure temperature per request
        generation_config = {
            "temperature": req.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 768,
        }

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt, generation_config=generation_config)

        answer_text = resp.text if hasattr(resp, "text") and resp.text else ""
        if not answer_text:
            answer_text = "I couldn’t find this in the course catalog."

        return AskResponse(model=GEMINI_MODEL_NAME, answer=answer_text, courses=courses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
