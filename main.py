from dotenv import load_dotenv
load_dotenv()

import os
import logging
import re
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import AsyncGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent API",
    description="A production-ready AI agent API.",
    version="1.0.0"
)

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

class SolveRequest(BaseModel):
    query: str = Field(..., description="The question to answer")
    assets: Optional[List[str]] = Field(default=[])

class SolveResponse(BaseModel):
    output: str

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):

    query = request.query.lower()

    # 🔥 RULE-BASED SOLUTIONS (BOOST ACCURACY)

    # Addition
    match = re.search(r'(\d+)\s*\+\s*(\d+)', query)
    if match:
        return SolveResponse(output=f"The sum is {int(match.group(1)) + int(match.group(2))}.")

    # Subtraction
    match = re.search(r'(\d+)\s*-\s*(\d+)', query)
    if match:
        return SolveResponse(output=f"The difference is {int(match.group(1)) - int(match.group(2))}.")

    # Multiplication
    match = re.search(r'(\d+)\s*\*\s*(\d+)', query)
    if match:
        return SolveResponse(output=f"The product is {int(match.group(1)) * int(match.group(2))}.")

    # Division
    match = re.search(r'(\d+)\s*/\s*(\d+)', query)
    if match:
        b = int(match.group(2))
        if b != 0:
            return SolveResponse(output=f"The quotient is {int(match.group(1)) // b}.")

    # Factorial
    match = re.search(r'factorial of (\d+)', query)
    if match:
        n = int(match.group(1))
        fact = 1
        for i in range(1, n+1):
            fact *= i
        return SolveResponse(output=f"The factorial is {fact}.")

    try:
        # 🤖 LLM fallback for complex queries
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You answer every question in exactly one short sentence ending with a period.

For math:
- Addition: The sum is X.
- Subtraction: The difference is X.
- Multiplication: The product is X.
- Division: The quotient is X.
- Factorial: The factorial is X.

Rules:
- ALWAYS end with a period.
- NO extra words.
- NO explanation.
- Keep it extremely concise.
"""
                },
                {
                    "role": "user",
                    "content": request.query
                }
            ],
            temperature=0,
            max_tokens=50
        )

        content = response.choices[0].message.content.strip()

        # 🔒 FINAL SAFETY: ensure period
        if not content.endswith("."):
            content += "."

        return SolveResponse(output=content)

    except Exception as e:
        logger.error(f"Error: {e}")
        return SolveResponse(output="Could not process accurately.")