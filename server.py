import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("No GEMINI_API_KEY found in .env file!")

# 2. CONFIGURE GEMINI
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-pro")

# 3. LOAD PORTFOLIO DATA
try:
    with open("portfolio.json", "r") as f:
        knowledge_base = json.load(f)
        context_data = json.dumps(knowledge_base, indent=2)
except FileNotFoundError:
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
    4. If the info isn't there, say "I don't have that info, but you can contact me directly."

    USER QUESTION: {request.question}
    """

    try:
        # We add a simple safety setting to prevent blocking innocent questions
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        # Print the exact error to the terminal so we can see it
        print(f"Gemini API Error: {e}")
        return {"answer": "I'm currently overloaded (Error 429). Please try again in a minute!"}