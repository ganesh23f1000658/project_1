import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)
