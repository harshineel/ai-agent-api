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
                    "content": """You are a precise question answering assistant. Follow these rules STRICTLY:

1. YES/NO QUESTIONS (starts with Is, Are, Was, Were, Do, Does, Did, Can, Could, Will, Would, Has, Have, Had):
   → Reply ONLY 'YES' or 'NO' in capitals.
   Example: 'Is 9 an odd number?' → 'YES'

2. EXTRACTION QUESTIONS (extract, find, get, identify):
   → Return ONLY the extracted value. No extra words.
   Example: 'Extract date from: Meeting on 12 March 2024' → '12 March 2024'

3. NUMBER/MATH OPERATIONS ON A LIST (sum, average, count, max, min, filter):
   → Return ONLY the final number. Nothing else.
   Example: 'Numbers: 2,5,8,11. Sum even numbers.' → '10'
   Example: 'Numbers: 1,2,3,4,5. Count odd numbers.' → '3'

4. MATH QUESTIONS (two numbers with operation):
   → Return in format 'The sum/difference/product/quotient is X.'
   Example: 'What is 10 + 15?' → 'The sum is 25.'

5. FACTUAL QUESTIONS:
   → Answer in one short sentence ending with a period.
   Example: 'What is the capital of France?' → 'The capital of France is Paris.'

NEVER add explanations or extra text. Return ONLY what the format requires."""
                },
                {
                    "role": "user",
                    "content": request.query
                }
            ],
            temperature=0,
            max_tokens=50
        )

        content = response.choices[0].message.content or "Could not process accurately"
        return SolveResponse(output=content.strip())

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        return SolveResponse(output="Could not process accurately")
