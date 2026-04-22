from dotenv import load_dotenv
load_dotenv()

import os
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
                    "content": """You answer questions in exactly one short sentence ending with a period.
For math questions always use this exact style:
- Addition: 'The sum is X.'
- Subtraction: 'The difference is X.'
- Multiplication: 'The product is X.'
- Division: 'The quotient is X.'
- Factorial: 'The factorial is X.'
- Square root: 'The square root is X.'
For all other questions, answer in one clean sentence ending with a period.
Never add any extra words, explanation, or formatting."""
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