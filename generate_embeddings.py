import json
import os
import google.generativeai as genai

# Configure the Gemini API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ Please set the 'GEMINI_API_KEY' environment variable.")

genai.configure(api_key=api_key)

# Embedding function
def get_embedding(text):
    if not text or not text.strip():
        raise ValueError("❌ Cannot embed empty or whitespace-only text.")
    
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

# Load scraped data
with open("scraped_data.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)

embeddings = []

# Process each link
for topic in scraped_data:
    for link in topic.get("content", []):
        text = link.get("text", "").strip()
        if not text:
            print(f"⚠️ Skipping empty text in link: {link.get('url', 'unknown')}")
            continue
        try:
            embedding = get_embedding(text)
            embeddings.append({
                "url": link.get("url", ""),
                "text": text,
                "embedding": embedding
            })
        except Exception as e:
            print(f"❌ Failed to embed: {link.get('url', 'unknown')} – {e}")

# Save embeddings to JSON
with open("scraped_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2)

print("✅ Embeddings saved to 'scraped_embeddings.json'.")

