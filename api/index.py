from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from PIL import Image
from io import BytesIO
import os
import json
import google.generativeai as genai

# Load Gemini key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

# Load scraped data once
with open("scraped_data.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)

class QuestionRequest(BaseModel):
    question: str
    image: str = None  # Optional base64 image

def find_relevant_links(question: str):
    keywords = question.lower().split()
    matches = []
    for topic in scraped_data:
        for link in topic["content"]:
            if any(kw in link["text"].lower() for kw in keywords):
                matches.append(link)
    return matches[:3]  # Limit to top 3 links

@app.post("/")
async def answer_question(payload: QuestionRequest):
    if payload.image:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Vision also supported in 1.5
        image_data = base64.b64decode(payload.image)
        image = Image.open(BytesIO(image_data))
        response = model.generate_content([payload.question, image])
    else:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(payload.question)

    related_links = find_relevant_links(payload.question)

    return JSONResponse({
        "answer": response.text,
        "links": related_links
    })
