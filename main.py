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

def handle_number_list(query: str):
    q = query.lower()
    # Extract all numbers from the query
    numbers = re.findall(r'-?\d+\.?\d*', query)
    if not numbers:
        return None
    nums = [float(n) for n in numbers]

    # Sum even numbers
    if 'sum even' in q or 'sum of even' in q:
        result = sum(n for n in nums if n % 2 == 0)
        return str(int(result) if result.is_integer() else result)

    # Sum odd numbers
    if 'sum odd' in q or 'sum of odd' in q:
        result = sum(n for n in nums if n % 2 != 0)
        return str(int(result) if result.is_integer() else result)

    # Count even
    if 'count even' in q or 'how many even' in q:
        return str(len([n for n in nums if n % 2 == 0]))

    # Count odd
    if 'count odd' in q or 'how many odd' in q:
        return str(len([n for n in nums if n % 2 != 0]))

    # Average/mean
    if 'average' in q or 'mean' in q:
        result = sum(nums) / len(nums)
        return str(int(result) if result.is_integer() else round(result, 2))

    # Max
    if 'max' in q or 'largest' in q or 'greatest' in q:
        return str(int(max(nums)) if max(nums).is_integer() else max(nums))

    # Min
    if 'min' in q or 'smallest' in q or 'lowest' in q:
        return str(int(min(nums)) if min(nums).is_integer() else min(nums))

    # Sum all
    if 'sum' in q or 'total' in q or 'add' in q:
        result = sum(nums)
        return str(int(result) if result.is_integer() else result)

    return None

def handle_yes_no(query: str):
    q = query.lower().strip()
    yes_no_starters = ('is ', 'are ', 'was ', 'were ', 'do ', 'does ',
                       'did ', 'can ', 'could ', 'will ', 'would ',
                       'has ', 'have ', 'had ')
    if q.startswith(yes_no_starters):
        return "YES_NO"
    return None

def handle_extraction(query: str):
    q = query.lower()
    if 'extract' in q or 'find the date' in q or 'what is the date' in q:
        return "EXTRACT"
    return None

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):
    try:
        query = request.query

        # Try number list operations first (pure code)
        number_result = handle_number_list(query)
        if number_result:
            return SolveResponse(output=number_result)

        # Detect question type for LLM prompt
        question_type = handle_yes_no(query) or handle_extraction(query) or "GENERAL"

        if question_type == "YES_NO":
            system_prompt = "Answer ONLY with 'YES' or 'NO' in capitals. Nothing else."
        elif question_type == "EXTRACT":
            system_prompt = "Extract and return ONLY the requested value. No extra words."
        else:
            system_prompt = """Answer in one short sentence ending with a period.
For math: 'The sum is 25.' For facts: 'The capital of France is Paris.'
No extra explanation."""

        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=50
        )

        content = response.choices[0].message.content or "Could not process accurately"
        return SolveResponse(output=content.strip())

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        return SolveResponse(output="Could not process accurately")
