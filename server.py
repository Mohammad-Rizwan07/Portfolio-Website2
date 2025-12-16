import os
import json
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("No GEMINI_API_KEY found in .env file!")

genai.configure(api_key=API_KEY)

# 2. SELECT THE MODEL (With Backup)
# Priority: Use free-tier models with better quota management
# 1. Try Gemma-3 (best for free tier)
# 2. Fallback to gemini-2.0-flash-lite (lower quota usage)
# 3. Fallback to gemini-1.5-flash
model = None
model_priority = [
    ('gemma-3-12b-it', 'Gemma 3 12B (Free, efficient)'),
    ('gemini-2.0-flash-lite', 'Gemini 2.0 Flash Lite'),
    ('gemini-1.5-flash', 'Gemini 1.5 Flash'),
    ('gemini-pro', 'Gemini Pro'),
]

for model_name, description in model_priority:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"[OK] Using model: {description}")
        break
    except Exception as e:
        print(f"[INFO] {model_name} not available: {str(e)[:50]}")
        continue

if not model:
    raise ValueError("[ERROR] Critical Error: No available models found!")

# 3. LOAD PORTFOLIO DATA
try:
    with open("portfolio.json", "r") as f:
        knowledge_base = json.load(f)
        context_data = json.dumps(knowledge_base, indent=2)
except FileNotFoundError:
    print("❌ Error: portfolio.json not found!")
    context_data = "No data available."

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
async def health_check():
    """Check if the server and API are working"""
    try:
        # List available models
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {
            "status": "✅ OK",
            "api_key_configured": bool(API_KEY),
            "current_model": str(model._client_config.api_key is not None),
            "available_models": models
        }
    except Exception as e:
        return {
            "status": "❌ ERROR",
            "error": str(e)
        }

@app.post("/ask")
async def ask_gemini(request: QueryRequest):
    print(f"Incoming Question: {request.question}")

    prompt = f"""
    You are an AI assistant for a developer's portfolio website.
    Use the following JSON data to answer the user's question.
    
    DATA:
    {context_data}

    RULES:
    1. Answer ONLY based on the data provided.
    2. Keep answers professional but friendly.
    3. Keep answers concise (max 3 sentences).
    
    USER QUESTION: {request.question}
    """

    try:
        response = model.generate_content(prompt)
        answer = response.text
        print(f"[OK] Response generated: {answer[:50]}...")
        return {"answer": answer}
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Gemini API Error: {type(e).__name__}: {error_msg[:100]}")
        
        # QUOTA EXCEEDED - Return helpful message
        if "exceeded" in error_msg.lower() or "quota" in error_msg.lower():
            return {
                "answer": "API quota exceeded. The free tier limit has been reached. This will reset in 24 hours, or you can upgrade your billing plan at https://console.cloud.google.com/billing for continued use."
            }
        
        return {"answer": f"I'm having trouble processing your question. Please try again later."}