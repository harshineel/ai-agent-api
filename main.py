from dotenv import load_dotenv
load_dotenv()

import os
import re
import math
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import AsyncGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent API", version="1.0.0")

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

class SolveRequest(BaseModel):
    query: str = Field(..., description="The question to answer")
    assets: Optional[List[str]] = Field(default=[], description="Asset URLs")

class SolveResponse(BaseModel):
    output: str

def is_yes_no_question(query: str) -> bool:
    q = query.lower().strip()
    return q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "will ", "would ", "has ", "have ", "had "))

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise information extraction and question answering assistant.

Rules:
- If the question is a YES/NO question (starts with Is, Are, Was, Were, Do, Does, Did, Can, Could, Will, Would, Has, Have, Had), reply with ONLY 'YES' or 'NO' in capitals.
- If asked to extract something (date, name, number, etc.), return ONLY the extracted value. Nothing else.
- If asked a math question, return ONLY in format like 'The sum is 25.'
- If asked a factual question, answer in one short sentence ending with a period.
- NEVER add extra words or explanation.

Examples:
- 'Is 9 an odd number?' → 'YES'
- 'Is Paris the capital of Germany?' → 'NO'
- 'Extract date from: Meeting on 12 March 2024' → '12 March 2024'
- 'What is 10 + 15?' → 'The sum is 25.'
- 'What is the capital of France?' → 'The capital of France is Paris.'"""
                },
                {
                    "role": "user",
                    "content": request.query
                }
            ],
            temperature=0,
            max_tokens=100
        )

        content = response.choices[0].message.content or "Could not process accurately"
        return SolveResponse(output=content.strip())

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        return SolveResponse(output="Could not process accurately")