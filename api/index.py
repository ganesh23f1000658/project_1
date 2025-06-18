from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
import os
import json
import numpy as np
from numpy.linalg import norm
import google.generativeai as genai

# Setup Gemini API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ GEMINI_API_KEY not found")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings
base_path = os.path.dirname(__file__)
cached_embeddings = []
try:
    with open(os.path.join(base_path, "..", "scraped_embeddings.json"), "r") as f:
        cached_embeddings = json.load(f)
except Exception as e:
    print("⚠️ Failed to load embeddings:", e)

# Request model
class QuestionRequest(BaseModel):
    question: str
    image: str | None = None

# Embedding logic
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))

def get_embedding(text):
    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return res["embedding"]
    except Exception as e:
        print("❌ Embedding failed:", e)
        return None

def find_relevant_context(question, top_k=5):
    q_embed = get_embedding(question)
    if not q_embed or not cached_embeddings:
        return []
    scored = [(cosine_similarity(q_embed, e["embedding"]), e) for e in cached_embeddings]
    return [e for _, e in sorted(scored, key=lambda x: -x[0])[:top_k]]

# Endpoint
@app.post("/")
async def ask(payload: QuestionRequest):
    top_context = find_relevant_context(payload.question)
    links = [{"url": i["url"], "text": i["text"]} for i in top_context]
    context_text = "\n".join(i["text"] for i in top_context)

    prompt = (
        "You are a helpful assistant for Tools in Data Science (TDS).\n\n"
        f"Question: {payload.question}\n\n"
        f"Relevant Notes:\n{context_text}"
    )

    try:
        if payload.image:
            image = Image.open(BytesIO(base64.b64decode(payload.image)))
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)

        return JSONResponse({
            "answer": response.text,
            "links": links
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "answer": "Generation failed.", "links": links}
        )
