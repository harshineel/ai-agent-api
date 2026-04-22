from dotenv import load_dotenv
load_dotenv()

import os
import re
import logging
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
    assets: Optional[List[str]] = Field(default=[], description="Asset URLs")

class SolveResponse(BaseModel):
    output: str

def detect_math(query: str):
    q = query.lower()
    numbers = re.findall(r'\d+\.?\d*', query)
    if len(numbers) >= 2:
        a, b = float(numbers[0]), float(numbers[1])
        if any(w in q for w in ['sum', 'add', 'plus', '+']):
            result = int(a + b) if (a + b).is_integer() else a + b
            return f"The sum is {result}."
        if any(w in q for w in ['subtract', 'minus', 'difference', '-']):
            result = int(a - b) if (a - b).is_integer() else a - b
            return f"The difference is {result}."
        if any(w in q for w in ['multiply', 'product', 'times', '*', 'x']):
            result = int(a * b) if (a * b).is_integer() else a * b
            return f"The product is {result}."
        if any(w in q for w in ['divide', 'quotient', '/']):
            result = int(a / b) if (a / b).is_integer() else round(a / b, 2)
            return f"The quotient is {result}."
    return None

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):

    try:
        # Try direct math detection first
        math_result = detect_math(request.query)
        if math_result:
            return SolveResponse(output=math_result)

        # Fall back to LLM for non-math questions
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Answer every question in exactly one short sentence ending with a period.
Be direct and concise. No extra explanation or formatting."""
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
        return SolveResponse(output=content)

    except Exception as e:
        logger.error(f"Error processing request: {type(e).__name__}: {e}")
        return SolveResponse(output="Could not process accurately")